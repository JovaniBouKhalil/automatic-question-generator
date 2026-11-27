"""Development pilot: same saved targets/drafts, explicit answer roles, optional draft removal."""
import hashlib
import json
import re
import time
from pathlib import Path

from questionforge.learning.pipeline import EDIT_SYSTEM, question_issue
from questionforge.learning.quality import QUALITY_MODEL, quality_chat_json
from questionforge.text import LANGUAGES

FOCUS_SYSTEM = EDIT_SYSTEM + """
Before writing, identify the role of the exact target answer in its evidence.
Follow answer_focus: the question must request that role, rather than the other endpoint
of the relation. Treat a faulty draft as untrusted; rewrite it completely when necessary.
For Arabic purpose answers, use a natural purpose question such as 'ما الغرض من استخدام ...؟'.
Keep Arabic active/passive roles and French agreement correct. The source determines the subject.
Do not copy an answer-focus hint or its placeholders into the question."""

def focus(fact):
    evidence,answer=fact['evidence'],fact['answer']
    pos=evidence.find(answer)
    before=evidence[:pos].strip() if pos>=0 else ''
    if re.search(r'\bwhile\b|\btandis que\b|بينما',evidence,re.I) and re.search(r'\bless\b|\bmore\b|\bmoins\b|davantage|أقل|أكبر',evidence,re.I):
        return 'Explicit comparison: ask how the stated conditions or outcomes differ. The target describes the contrast, not its explanation; do not ask a why-question for this target.'
    if re.search(r'(?:convert\w*|يحول|تحول).+(?:into|en|إلى)$',before,re.I):
        return 'Conversion output: ask what the source device converts its input into. The target names the resulting output, not the converting device.'
    if fact['question_type']=='cause_effect':
        return 'Explicit reason or prevention purpose: ask why the stated action is taken. Preserve negation and the action as described.'
    if re.search(r'(?:used to|used for|designed to|purpose .{0,50} was to|servent à|sert à|servait à|permet de)$',before,re.I) or re.search(r'(?:تستخدم|يستخدم|تستعمل|يستعمل).+\sل$',before) or re.search(r'(?:الهدف من|الغرض من)',before):
        return 'Intended use or purpose: ask what the thing/data is used for. Do not phrase the data as an agent using another object. Arabic pattern: ما الغرض من استخدام ...؟'
    if re.search(r'(?:does not preserve|ne conserve pas|لا يحتفظ ب)$',before,re.I):
        return 'Missing information: ask which information/property is NOT retained. Preserve the negative relation and name the representation.'
    if re.search(r'(?:records|enregistre|يسجل)$',before,re.I):
        return 'Recorded information: ask what information the representation records, rather than what records that information.'
    if re.search(r'(?:بحسب|according to|selon)$',before,re.I):
        return 'Criterion: ask on what basis the described adjustment is made.'
    return 'Ask for the specific stated fact supplied as the target; keep subjects, predicates, scope and exceptions faithful.'

def main():
    ready=Path('results/learning_focus_dev_4b_v1/summary.json')
    print('Waiting for both original 4B comparisons to finish before the separate pilot.',flush=True)
    while not ready.exists() or not json.loads(ready.read_text(encoding='utf-8')).get('study_complete'):
        time.sleep(10)
    output=Path('results/quality/learning_4b_role_focus_pilot.json')
    if output.exists():raise ValueError('Preserve the existing pilot; do not overwrite.')
    report={'model':QUALITY_MODEL.model_id,'revision':QUALITY_MODEL.revision,'precision':'none-float32',
            'scope':'Purposive development pilot, not a final test or human usefulness assessment. Hinted editing versus draft-free hinted writing share exact targets/context.',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'system':FOCUS_SYSTEM,'cases':[]}
    for name in ['learning_light_and_voltage_ar','learning_training_and_evaluation_ar','learning_light_and_voltage_en','learning_text_representation_en','learning_training_and_evaluation_fr','learning_text_representation_fr']:
        path=Path('results/learning_focus_educational_4b_v1')/(name+'.json')
        case=json.loads(path.read_text(encoding='utf-8'))
        plan=case['planning_report']
        entry={'source_case':path.as_posix(),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
               'context':case['context'],'facts':plan['facts'],'baseline_raw':plan['raw_edit'],'variants':[]}
        for variant in ['role_hinted_edit','role_hinted_draft_free']:
            payload={'language':LANGUAGES[case['language']],'learning_objective':plan['learning_objective'],
                     'passage':case['context'],'facts':[{**f,'index':i,'answer_focus':focus(f)} for i,f in enumerate(plan['facts'])]}
            if variant=='role_hinted_edit':
                for i,f in enumerate(payload['facts']):f['draft_question']=plan['drafts'][i]['question']
            start=time.monotonic()
            row={'variant':variant,'payload':payload}
            print('Starting',name,variant,flush=True)
            try:
                parsed,raw=quality_chat_json(FOCUS_SYSTEM,payload)
                row.update(raw=raw,parsed=parsed)
                row['format_issues']=[question_issue(item.get('question'),plan['facts'][item['index']]['answer'],case['language']) for item in parsed]
            except Exception as exc:row.update(error=f'{type(exc).__name__}: {exc}',raw=getattr(exc,'raw_text',None))
            row['seconds']=time.monotonic()-start
            entry['variants'].append(row)
            report['cases']=[*report['cases'][:len(report['cases'])-1],entry] if report['cases'] and report['cases'][-1]['source_case']==entry['source_case'] else [*report['cases'],entry]
            output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(row,ensure_ascii=False),flush=True)
    print('Role-focus pilot complete; semantic review required.',flush=True)

if __name__=='__main__':main()
