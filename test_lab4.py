import ast
import json
from pathlib import Path
import sys
import types

import pytest
from pydantic import ValidationError


ROOT = Path(__file__).resolve().parent
NOTEBOOK = json.loads((ROOT / 'lab4.ipynb').read_text(encoding='utf-8'))
MODULE = types.ModuleType('lab4_notebook')
sys.modules[MODULE.__name__] = MODULE
IMPORTS = ast.parse(''.join(NOTEBOOK['cells'][0]['source']))
IMPORTS.body = [node for node in IMPORTS.body if
               isinstance(node, ast.Import) and all(item.name in {'json'} for item in node.names)
               or isinstance(node, ast.ImportFrom) and node.module == 'pydantic']
exec(compile(IMPORTS, 'lab4.ipynb:imports', 'exec'), MODULE.__dict__)
for cell in NOTEBOOK['cells']:
    if 'core' in cell['metadata'].get('tags', []):
        exec(compile(''.join(cell['source']), f"lab4.ipynb:{cell['id']}", 'exec'), MODULE.__dict__)

CONFIG = json.loads((ROOT / 'generation_config.json').read_text(encoding='utf-8'))
PROMPTS = json.loads((ROOT / 'prompts.json').read_text(encoding='utf-8'))
ROWS = json.loads((ROOT / 'semantic_checks.json').read_text(encoding='utf-8'))
EMPTY = {field: [] for field in MODULE.ALL_FIELDS}


@pytest.mark.parametrize('row', [pytest.param(row, id=row['case']) for row in ROWS])
def test_recorded_semantic_case(row):
    if not row['text'].strip():
        run = MODULE.run_one(row['text'], PROMPTS['version_2'],
                             lambda prompt: pytest.fail('Empty text must not call the model'),
                             'version_2', CONFIG)
        assert run['result'] == EMPTY and run['attempts'] == []
        assert row['result'] == EMPTY and row['attempts'] == []
        return
    check = MODULE.assess_response(row['attempts'][-1]['raw_response'], row['text'])
    assert MODULE.semantic_passed(check['result'], row['field'], row['expected'])


def test_extra_field_is_rejected():
    with pytest.raises(ValidationError):
        MODULE.validate_response(json.dumps(EMPTY | {'extra': 'value'}), '')


def test_absent_evidence_is_rejected():
    with pytest.raises(ValueError, match='Evidence is absent'):
        MODULE.validate_response(json.dumps(EMPTY | {'evidence': ['absent']}), '')


def test_invalid_response_gets_one_corrective_request():
    responses = iter(['not json', json.dumps(EMPTY)])
    calls = []

    def generate(prompt):
        calls.append(prompt)
        return next(responses), {}

    run = MODULE.run_one('Текст отзыва.', PROMPTS['version_2'], generate, 'version_2', CONFIG)
    assert run['valid'] and len(run['attempts']) == len(calls) == 2
    assert run['attempts'][0]['error'] in calls[1]


def test_two_invalid_responses_stop_generation():
    calls = []

    def generate(prompt):
        calls.append(prompt)
        return 'not json', {}

    run = MODULE.run_one('Текст отзыва.', PROMPTS['version_2'], generate, 'version_2', CONFIG)
    assert not run['valid'] and run['result'] is None
    assert len(run['attempts']) == len(calls) == 2


def test_valid_response_does_not_repeat():
    calls = []

    def generate(prompt):
        calls.append(prompt)
        return json.dumps(EMPTY), {}

    run = MODULE.run_one('Текст отзыва.', PROMPTS['version_2'], generate, 'version_2', CONFIG)
    assert run['valid'] and len(run['attempts']) == len(calls) == 1
