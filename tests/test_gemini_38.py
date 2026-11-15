import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from google.genai import live, types

from main import app
from sdk.core.model_registry import DEFAULT_MODEL_ID, resolve
from sdk.providers.gemini_live import _live_config, _read_turn, _responses, _track_interaction

LIVE = 'gemini-3.8-live'
THINKING = 'gemini-3.8-live-extended-thinking'


def session_for(*messages, model=THINKING):
    socket = SimpleNamespace(recv=AsyncMock(side_effect=[json.dumps(m) for m in messages]))
    session = live.AsyncSession(SimpleNamespace(vertexai=False), socket)
    _track_interaction(session, model)
    return session


def utterance(text, status):
    return {'serverContent': {'outputTranscription': {'text': text},
                              'turnComplete': True, 'interactionStatus': status}}


class ConfigTests(unittest.TestCase):
    def test_model_specific_thinking_and_tools(self):
        original = types.Tool(function_declarations=[types.FunctionDeclaration(name='lookup')])
        for budget, level in [(0, 'LOW'), (512, 'LOW'), (2048, 'MEDIUM'), (8192, 'HIGH')]:
            config = _live_config('test', tools=[original], model=THINKING, thinking_budget=budget)
            self.assertEqual(config.thinking_config.thinking_level, level)
            self.assertIsNone(config.thinking_config.thinking_budget)
            self.assertEqual(config.tools[0].function_declarations[0].behavior, 'NON_BLOCKING')
        self.assertIsNone(original.function_declarations[0].behavior)
        config = _live_config('test', model=LIVE, tools=[original], thinking_budget=8192)
        self.assertIsNone(config.thinking_config)
        self.assertEqual(config.tools[0].function_declarations[0].behavior, 'BLOCKING')
        self.assertIsNone(_live_config('test', model=THINKING).thinking_config)
        self.assertEqual(_live_config('test', thinking_budget=512).thinking_config.thinking_budget, 512)
        for model in (LIVE, THINKING):
            self.assertEqual(resolve(model + ':latest'), model)
        self.assertEqual(resolve('unknown'), DEFAULT_MODEL_ID)


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_filler_does_not_close_interaction(self):
        session = session_for(utterance('Checking. ', 'IN_PROGRESS'), utterance('Result.', 'IDLE'))
        text, calls, _ = await _read_turn(session)
        self.assertEqual(text, 'Checking. Result.')
        self.assertEqual(calls, [])
        self.assertEqual(session._ws.socket.recv.await_count, 2)

    async def test_async_tool_call_and_continuation(self):
        session = session_for(
            utterance('Checking. ', 'IN_PROGRESS'),
            {'toolCall': {'functionCalls': [{'id': 'call_1', 'name': 'lookup', 'args': {}}]},
             'interactionStatus': 'IN_PROGRESS'},
            utterance('Done.', 'IDLE'),
        )
        text, calls, _ = await _read_turn(session)
        self.assertEqual(text, 'Checking. ')
        self.assertEqual(calls[0].name, 'lookup')
        text, calls, _ = await _read_turn(session)
        self.assertEqual(text, 'Done.')
        self.assertEqual(calls, [])

    async def test_status_only_events_and_stream(self):
        session = session_for({'interactionStatus': 'IN_PROGRESS'},
                              utterance('Done.', 'IN_PROGRESS'), {'interactionStatus': 'IDLE'})
        chunks = [m.server_content.output_transcription.text async for m in _responses(session)
                  if m.server_content and m.server_content.output_transcription]
        self.assertEqual(chunks, ['Done.'])

    async def test_legacy_turn_complete(self):
        session = session_for({'serverContent': {'outputTranscription': {'text': 'Done.'},
                                                'turnComplete': True}}, model=DEFAULT_MODEL_ID)
        text, calls, _ = await _read_turn(session)
        self.assertEqual(text, 'Done.')

    async def test_models_visible_through_both_protocols(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            listed = (await client.get('/openai/v1/models')).json()['data']
            tags = (await client.get('/api/tags')).json()['models']
            for model in (LIVE, THINKING):
                self.assertIn(model, [m['id'] for m in listed])
                self.assertIn(model + ':latest', [m['name'] for m in tags])
                detail = await client.get('/openai/v1/models/' + model)
                self.assertEqual(detail.status_code, 200)
                shown = (await client.post('/api/show', json={'model': model + ':latest'})).json()
                self.assertEqual(shown['model_info']['general.basename'], model)
                self.assertIn('tools', shown['capabilities'])


if __name__ == '__main__':
    unittest.main()
