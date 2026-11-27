import json,time
from pathlib import Path
from questionforge.learning.pipeline import load_planner, question_issue
from questionforge.learning.rules import focused_evidence

tokenizer,model,torch,device=load_planner()
results=[]
probes=[('learning_text_representation_ar',i) for i in range(3)]+[('learning_light_and_voltage_ar',0)]
for name,index in probes:
    case=json.loads((Path('results/learning_focus_educational_v3')/(name+'.json')).read_text(encoding='utf-8'))
    fact=case['planning_report']['facts'][index]
    context,_=focused_evidence(case['context'],fact['answer'],fact['answer_start'],'ar')
    system='أنت معلم لغة عربية. اكتب سؤال فهم واحداً تدعمه معلومات النص وتكون إجابته العبارة المطلوبة. لا تكتب الإجابة أو أي شرح. لا تفترض معلومات غير مذكورة.'
    user=f"النص:\n{context}\n\nالإجابة المطلوبة:\n{fact['answer']}\n\nاكتب السؤال فقط، ولا تذكر الإجابة في السؤال."
    encoded=tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':user}],tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(device)
    start=time.monotonic()
    with torch.inference_mode():out=model.generate(**encoded,do_sample=False,max_new_tokens=80,pad_token_id=tokenizer.pad_token_id)
    raw=tokenizer.decode(out[0][encoded['input_ids'].shape[1]:],skip_special_tokens=True)
    row={'case':name,'fact_index':index,'system':system,'user':user,'answer':fact['answer'],'raw':raw,'format_issue':question_issue(raw,fact['answer'],'ar'),'seconds':time.monotonic()-start}
    results.append(row)
    Path('results/quality/learning_plain_arabic_pilot.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(row,ensure_ascii=False),flush=True)
