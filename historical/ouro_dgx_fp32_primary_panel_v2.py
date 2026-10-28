#!/usr/bin/env python3
"""Frozen primary-alpha Ouro loop-boundary precision study on DGX Spark.

Two modes: --mode preflight (first frozen question, exhaustive three-arm panel,
with exact all-condition, batch-matched original-hook parity) and --mode full (all 30; resume-safe).

V2 fixes the batch-size mismatch that failed the v1 preflight.
Every frozen FP16 condition must match the original hook exactly (all chunks).

The original project is read-only. All outputs go under /workspace/dgx_validation/.
The machine does NOT reproduce the RTX4060 frozen FP16 activations exactly;
archived Windows-vs-DGX differences are reported, never used as an equivalence gate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from datasets import Dataset
from lm_eval.models.huggingface import HFLM
import run_boundary_causal_evaluation as frozen

ROOT = Path('/workspace/ouro')
OUT = Path('/workspace/dgx_validation/fp32_robustness')
ALPHA = 0.08
ARMS = ('dgx_fp16', 'dgx_fp32_realized', 'dgx_fp32_intended')
BOUNDS = (2, 3)
LABELS = ('seed42_feature2913', 'seed123_feature2437')
BAND = 0.005  # Astra-proposed new numerical-equivalence threshold; not preregistered.
REPS = 20000
SEED = 20261030
SCRIPT_VERSION = 'ouro_dgx_fp32_primary_panel_v2'
HASH_EVALUATOR = 'a7872dabcfbdaac54dd3082e82da25b1ed98f7f4fac34b838efb0d8e67e77693'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def checkpoint_write(path, margins, completed, identity):
    tmp = path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, margins=margins, completed_questions=np.int64(completed),
                            signature=np.array(identity))
    os.replace(tmp, path)


def build_identity(args, hashes, row_ids):
    return {
        'protocol': SCRIPT_VERSION, 'alpha': ALPHA, 'boundaries': list(BOUNDS),
        'directions': list(LABELS), 'controls_per_direction': 16,
        'arms': list(ARMS), 'batch_size': args.batch_size,
        'row_ids': row_ids, 'hashes': hashes,
        'script_sha256': digest(Path(__file__)),
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'platform': platform.machine(),
        'python': sys.version.split()[0],
        'fp32_weights': 'copy.deepcopy of DGX-loaded FP16 model, then float() conversion',
        'base_states': 'DGX FP16 complete-sequence states, captured separately at the exact intervention-chunk batch sizes',
        'cross_hardware_exact_reproduction': False,
        'comparison_band_T_units': [-BAND, BAND],
        'comparison_band_origin': 'Astra-proposed numerical criterion; NOT original pilot preregistration',
        'bootstrap': {'seed': SEED, 'draws': REPS, 'sampling_unit': 'frozen question'},
    }


def load_frozen():
    paths = {
        'dataset': ROOT / 'hf_cache/datasets/allenai___sciq/default/0.0.0/2c94ad3e1aafab77146f384e23536f97a4849815/sciq-validation.arrow',
        'model_source': ROOT / 'hf_cache/modules/transformers_modules/ouro_model/modeling_ouro.py',
        'controls': ROOT / 'results/boundary_pilot_matched_controls_v1.pt',
        'baseline': ROOT / 'results/boundary_pilot_baseline_states_v1.pt',
        'contrasts': ROOT / 'results/boundary_pilot_contrasts_v1.json',
        'model_weights': ROOT / 'ouro_model/model.safetensors',
        'original_evaluator': ROOT / 'run_boundary_causal_evaluation.py',
        'original_npz': ROOT / 'results/boundary_pilot_causal_evaluation_v1.npz',
    }
    expected = {
        'dataset': frozen.EXPECTED_DATA_ARROW_SHA256,
        'model_source': frozen.EXPECTED_MODELING_SOURCE_SHA256,
        'controls': frozen.EXPECTED_CONTROL_BANK_SHA256,
        'baseline': frozen.EXPECTED_BASELINE_STATES_SHA256,
        'original_evaluator': HASH_EVALUATOR,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f'Missing frozen input {name}: {path}')
    hashes = {name: digest(path) for name, path in paths.items()}
    for name, value in expected.items():
        if hashes[name] != value:
            raise RuntimeError(f'Frozen SHA mismatch {name}: {hashes[name]}')
    print('PASS pinned hashes: dataset, model source, control bank, baseline, original evaluator', flush=True)
    records = frozen.load_records(paths['contrasts'])
    evaluation = [r for r in records if str(r.get('role','')).lower() == 'evaluation'
                  and bool(r.get('contrast_eligible',False))]
    if len(evaluation) != 30:
        raise RuntimeError(f'Frozen panel size {len(evaluation)} != 30')
    row_ids = [str(r['row_id']) for r in evaluation]
    if len(set(row_ids)) != 30:
        raise RuntimeError('Repeated frozen row IDs')
    ctrl_payload = frozen.torch_load_local(paths['controls'])
    labels = tuple(str(x) for x in ctrl_payload['direction_labels'])
    if labels != LABELS or [str(x) for x in ctrl_payload['evaluation_row_ids']] != row_ids:
        raise RuntimeError('Frozen control labels or row ordering changed')
    directions = torch.as_tensor(ctrl_payload['candidate_directions'], dtype=torch.float32).cpu()
    controls = torch.as_tensor(ctrl_payload['controls'], dtype=torch.float32).cpu()
    if directions.shape != (2,2048) or controls.shape != (30,2,16,2048):
        raise RuntimeError('Frozen vector shapes changed')
    if not torch.isfinite(directions).all() or not torch.isfinite(controls).all():
        raise RuntimeError('Nonfinite frozen vector')
    data = Dataset.from_file(str(paths['dataset']))
    questions = []
    for record in evaluation:
        row = dict(data[int(record['source_row_index'])])
        context = frozen.get_context(row)
        if hashlib.sha256(context.encode('utf8')).hexdigest() != record['context_sha256']:
            raise RuntimeError(f'Frozen context mismatch: {record["row_id"]}')
        gold = int(record['gold_first_continuation_token_id'])
        wrong = int(record['incorrect_first_continuation_token_id'])
        if gold == wrong:
            raise RuntimeError('Identical answer tokens')
        questions.append({'row_id':str(record['row_id']), 'context':context,
                          'length':int(record['context_token_count']), 'gold':gold, 'wrong':wrong})
    npz = np.load(paths['original_npz'], allow_pickle=False)
    if list(npz['row_ids']) != row_ids or tuple(npz['direction_labels']) != LABELS:
        raise RuntimeError('Archived NPZ ordering changed')
    if not np.array_equal(npz['alphas'], [0.08,0.16]) or not np.array_equal(npz['boundaries'], [2,3]):
        raise RuntimeError('Archived NPZ axes changed')
    archive = np.array(npz['margins'], copy=True)
    if archive.shape != (30,2,2,17,2,2) or not np.isfinite(archive).all():
        raise RuntimeError('Archived margin array invalid')
    print('PASS 30 frozen prompts, 2 directions, 16 controls each, archive array ordering', flush=True)
    return questions, directions, controls, archive[:,:,:,:,0,:], hashes, row_ids


def set_fp32_ieee():
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
    if hasattr(torch.backends.cudnn, 'conv') and hasattr(torch.backends.cudnn.conv, 'fp32_precision'):
        torch.backends.cudnn.conv.fp32_precision = 'ieee'
    if torch.get_float32_matmul_precision() != 'highest':
        raise RuntimeError('FP32 highest precision not enabled')


def load_models():
    print('Loading frozen FP16 Ouro model', flush=True)
    lm = HFLM(pretrained=str(ROOT/'ouro_model'), device='cuda:0', dtype='float16',
              trust_remote_code=True, batch_size=1)
    m16 = lm.model.eval()
    m16.config.use_cache = m16.model.config.use_cache = False
    print('Creating FP32 continuation model by upcasting the FP16-loaded weights', flush=True)
    m32 = copy.deepcopy(m16).float().eval()
    m32.config.use_cache = m32.model.config.use_cache = False
    p16 = dict(m16.named_parameters())
    p32 = dict(m32.named_parameters())
    if p16.keys() != p32.keys():
        raise RuntimeError('Parameters differ between FP16 and FP32 models')
    for name in p16:
        a, b = p16[name].detach(), p32[name].detach()
        if a.is_floating_point() and (b.dtype != torch.float32 or not torch.equal(a.float(),b)):
            raise RuntimeError('FP32 parameter upcast identity failed: '+name)
    b16 = dict(m16.named_buffers())
    b32 = dict(m32.named_buffers())
    if b16.keys() != b32.keys():
        raise RuntimeError('FP32 buffer mismatch')
    for name in b16:
        a, b = b16[name], b32[name]
        if a.is_floating_point() and (b.dtype != torch.float32 or not torch.equal(a.float(),b)):
            raise RuntimeError('FP32 buffer upcast identity failed: '+name)
    print('PASS FP32 parameters and floating buffers = exact upcast of FP16 model', flush=True)
    return lm, m16, m32


def continue_logits(model, state, mask):
    frozen.set_loop_count(model,1)
    with torch.inference_mode(), torch.autocast('cuda',enabled=False):
        output, states, _ = model.model(inputs_embeds=state, attention_mask=mask, use_cache=False)
        if len(states)!=1:
            raise RuntimeError('Continuation execution count != 1')
        logits = model.lm_head(output.last_hidden_state[:,-1:,:])
    return logits


def direct_hook_batch(model, ids, mask, boundary, deltas):
    seen={'n':0}
    def hook(module, inputs, output):
        seen['n']+=1
        if seen['n'] == boundary:
            patched=output.clone()
            patched[:,-1,:]=(output[:,-1,:].float()+deltas).to(output.dtype)
            return patched
        return output
    handle=model.model.norm.register_forward_hook(hook)
    try:
        frozen.set_loop_count(model,boundary+1)
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            logits=model(input_ids=ids,attention_mask=mask,use_cache=False,
                         logits_to_keep=1,exit_at_step=boundary,return_dict=True).logits
    finally:
        handle.remove()
    if seen['n'] != boundary+1:
        raise RuntimeError(f'Original-hook norm calls {seen["n"]} != {boundary+1}')
    return logits


def all_vectors(directions, controls_q):
    # Variant index 0 is the candidate; 1..16 are frozen matched controls.
    return torch.stack([torch.cat((directions[d:d+1],controls_q[d]),dim=0)
                        for d in range(2)],dim=0)


def build_conditions(vectors):
    specs=[]
    deltas=[]
    for di in range(2):
        for vi in range(17):
            for signi,sign in enumerate((1,-1)):
                specs.append((di,vi,signi))
                # Frozen direction stored as FP32; same operation as validated smoke test.
                deltas.append(vectors[di,vi] * (ALPHA * sign))
    delta=torch.stack(deltas).to(device='cuda:0',dtype=torch.float32)
    if delta.shape != (68,2048) or not torch.isfinite(delta).all():
        raise RuntimeError('Unexpected frozen condition bank')
    return specs,delta


def score(logits,gold,wrong):
    v=logits[:,-1,:].float()
    margins=(v[:,gold]-v[:,wrong]).double().cpu().numpy()
    if not np.isfinite(margins).all():
        raise RuntimeError('Nonfinite answer margins')
    return margins


def run_question(qi, question, directions, control_bank, lm, m16, m32, batch_size):
    token_ids=frozen.as_python_int_list(lm.tok_encode(question['context']))
    if len(token_ids)!=question['length']:
        raise RuntimeError('Frozen token count mismatch '+question['row_id'])
    ids=torch.tensor([token_ids],device='cuda:0',dtype=torch.long)
    mask=torch.ones_like(ids)
    vectors=all_vectors(directions,control_bank[qi])
    specs,delta=build_conditions(vectors)
    result=np.full((2,2,17,3,2),np.nan,dtype=np.float64)

    # Critical v2 correction: CUDA FP16 output can depend on batch shape.
    # Capture the ENTIRE FP16 boundary state at each actual chunk batch size,
    # never broadcast a batch-1 state to batch-8 conditions. This is what the
    # frozen original-hook executor also sees for that batch size.
    batch_sizes=sorted({min(batch_size,68-start) for start in range(0,68,batch_size)})
    states_by_batch={}
    for b in batch_sizes:
        ids_b=ids.expand(b,-1).contiguous()
        mask_b=mask.expand(b,-1).contiguous()
        frozen.set_loop_count(m16,4)
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            _, full_states, _=m16.model(input_ids=ids_b,attention_mask=mask_b,
                                        use_cache=False)
        if len(full_states)!=4 or any(x.dtype!=torch.float16 for x in full_states):
            raise RuntimeError(f'Invalid DGX FP16 capture for batch size {b}')
        states_by_batch[b]={
            boundary:full_states[boundary-1].detach().clone()
            for boundary in BOUNDS
        }
        for boundary in BOUNDS:
            if states_by_batch[b][boundary].shape!=(b,len(token_ids),2048):
                raise RuntimeError('Unexpected batched full-sequence state shape')

    for bi,boundary in enumerate(BOUNDS):
        hook_comparisons=0
        for start in range(0,68,batch_size):
            stop=min(start+batch_size,68)
            b=stop-start
            d=delta[start:stop]
            masks=mask.expand(b,-1).contiguous()
            base=states_by_batch[b][boundary]
            st16=base.clone()
            st16[:,-1,:]=(base[:,-1,:].float()+d).to(torch.float16)
            st32_realized=st16.float()
            st32_intended=base.float().clone()
            st32_intended[:,-1,:]=base[:,-1,:].float()+d
            l16=continue_logits(m16,st16,masks)
            # Fail closed for EVERY condition, including the final partial batch.
            # A direct original-hook run must produce exactly the same logits as
            # the split continuation using the *matching* batched state.
            ids_b=ids.expand(b,-1).contiguous()
            direct=direct_hook_batch(m16,ids_b,masks,boundary,d)
            if not torch.equal(direct,l16):
                parity_max=(direct.float()-l16.float()).abs().max().item()
                raise RuntimeError(
                    f'FP16 ORIGINAL HOOK vs SPLIT mismatch: q={qi}, '
                    f'boundary={boundary}, chunk={start}:{stop}, batch={b}, '
                    f'max={parity_max}'
                )
            hook_comparisons += b
            l32r=continue_logits(m32,st32_realized,masks)
            l32i=continue_logits(m32,st32_intended,masks)
            if l16.dtype!=torch.float16 or l32r.dtype!=torch.float32 or l32i.dtype!=torch.float32:
                raise RuntimeError('Model output logits precision mismatch')
            scores=(score(direct,question['gold'],question['wrong']),
                    score(l32r,question['gold'],question['wrong']),
                    score(l32i,question['gold'],question['wrong']))
            for k,(di,vi,si) in enumerate(specs[start:stop]):
                for arm in range(3):
                    result[bi,di,vi,arm,si]=scores[arm][k]
            del st16,st32_realized,st32_intended,l16,direct,l32r,l32i
        if hook_comparisons!=68:
            raise RuntimeError('FP16 hook parity did not cover all 68 intervention conditions')
        print(f'  PASS boundary {boundary} FP16 original-hook parity: exact, '
              f'all {hook_comparisons} conditions across matching batch sizes',
              flush=True)
    if not np.isfinite(result).all():
        raise RuntimeError(f'Incomplete question results: {question["row_id"]}')
    return result


def paired(margins):
    # margins [question,boundary,direction,variant,arm,sign]
    s=(margins[...,0]-margins[...,1])/(2*ALPHA)
    d=s[:,1]-s[:,0]  # q,direction,variant,arm
    return d[:,:,0,:]-d[:,:,1:,:].mean(axis=2)  # q,direction,arm


def statistics(margins, archive, row_ids):
    paired_q=paired(margins)
    archived_s=(archive[...,0]-archive[...,1])/(2*ALPHA)
    archived_d=archived_s[:,1]-archived_s[:,0]
    archived_paired=archived_d[:,:,0]-archived_d[:,:,1:].mean(axis=2)
    rng=np.random.default_rng(SEED)
    draws=rng.integers(0,len(row_ids),size=(REPS,len(row_ids)))
    report={'scope':'frozen 30-question primary-alpha DGX robustness',
            'alpha':ALPHA,'candidate_directions':{},
            'proposed_equivalence_band_T_units':[-BAND,BAND],
            'band_is_new_post_pilot_proposal':True,
            'bootstrap_draws':REPS,'bootstrap_seed':SEED,
            'cross_hardware_archived_fp16_is_not_exact':True}
    all_pass=True
    for di,label in enumerate(LABELS):
        arm_means={arm:float(paired_q[:,di,ai].mean()) for ai,arm in enumerate(ARMS)}
        orig=float(archived_paired[:,di].mean())
        original_diff=paired_q[:,di,0]-archived_paired[:,di]
        comp={}
        for ai in (1,2):
            name=ARMS[ai]
            diff=paired_q[:,di,ai]-paired_q[:,di,0]
            replicates=diff[draws].mean(axis=1)
            ci=np.quantile(replicates,[0.025,0.975]).tolist()
            ci_family=np.quantile(replicates,[0.00625,0.99375]).tolist()
            passed=(ci[0]>-BAND and ci[1]<BAND)
            all_pass &= passed
            comp[name]={
                'mean_paired_T_discrepancy':float(diff.mean()),
                'bootstrap_95pct_CI':ci,
                'bootstrap_98p75pct_CI_4_comparison_familywise':ci_family,
                'inside_proposed_band_at_95pct':bool(passed),
                'inside_band_at_98p75pct':bool(ci_family[0]>-BAND and ci_family[1]<BAND),
                'per_question_RMS_discrepancy':float(np.sqrt(np.mean(diff**2))),
                'max_abs_per_question':float(np.abs(diff).max()),
                'directional_sign_changes_nonzero':int(np.sum(np.sign(paired_q[:,di,ai])!=np.sign(paired_q[:,di,0]))),
            }
        report['candidate_directions'][label]={
            'dgx_means_T':arm_means,
            'archived_windows_fp16_T':orig,
            'dgx_fp16_minus_archived_windows_fp16_T':float(original_diff.mean()),
            'dgx_fp16_vs_archived_windows_fp16_RMS_question_effect':float(np.sqrt(np.mean(original_diff**2))),
            'comparisons':comp,
        }
    report['all_four_95pct_CIs_wholly_inside_proposed_band']=bool(all_pass)
    report['scientific_interpretation']='New DGX hardware numerical assessment only; does not certify original Windows pilot or independent replication.'
    return report,paired_q,archived_paired


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('preflight','full'),required=True)
    p.add_argument('--batch-size',type=int,default=8)
    args=p.parse_args()
    if not 1<=args.batch_size<=8:
        raise RuntimeError('Batch size must be 1..8 in locked v1 program')
    print(f'=== {SCRIPT_VERSION} MODE={args.mode} ===',flush=True)
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0)!='NVIDIA GB10':
        raise RuntimeError('Expected DGX GB10 CUDA device')
    available=psutil.virtual_memory().available/1024**3
    print(f'Available unified RAM: {available:.2f} GiB',flush=True)
    if available < 24:
        raise RuntimeError('DGX unified memory safety gate: requires >=24 GiB available')
    set_fp32_ieee()
    questions,directions,control_bank,archive,hashes,row_ids=load_frozen()
    identity=build_identity(args,hashes,row_ids)
    OUT.mkdir(parents=True,exist_ok=True)
    pre_path=OUT/'fp32_primary_preflight_v2.json'
    full_identity=OUT/'fp32_primary_full_identity_v2.json'
    ckpt=OUT/'fp32_primary_full_checkpoint_v2.npz'
    final_npz=OUT/'fp32_primary_full_v2.npz'
    final_json=OUT/'fp32_primary_analysis_v2.json'
    if args.mode=='full':
        if not pre_path.is_file():
            raise RuntimeError('Run preflight first; abort full study')
        pre=json.loads(pre_path.read_text())
        if pre.get('status')!='PASS' or pre['identity']!=identity:
            raise RuntimeError('Preflight was not passed by this exact source/input/configuration')
        if final_npz.exists() or final_json.exists():
            raise RuntimeError('Final result exists: no silent overwrite')
        if full_identity.exists() and json.loads(full_identity.read_text()) != identity:
            raise RuntimeError('Locked full identity mismatches')
        if not full_identity.exists():
            atomic_json(full_identity,identity)
    print('Loading models...',flush=True)
    lm,m16,m32=load_models()
    if args.mode=='preflight':
        if pre_path.exists():
            raise RuntimeError('Preflight output already exists: no silent overwrite')
        print('PRE-FLIGHT: first frozen question, all 2x2x17x2 conditions, all three precision arms',flush=True)
        t=time.time()
        q=run_question(0,questions[0],directions,control_bank,lm,m16,m32,
                       args.batch_size)
        dgx_fp16_diff=q[:,:,:,0,:]-archive[0]
        report={'status':'PASS','identity':identity,
                'row_id':row_ids[0],
                'all_408_condition_margins_finite':True,
                'fp16_hook_split_batched_parity':'EXACT every condition for both boundaries, using batch-size-matched captured states',
                'dgx_fp16_vs_archived_windows_fp16_margin_max_abs':float(np.abs(dgx_fp16_diff).max()),
                'dgx_fp16_vs_archived_windows_fp16_margin_RMS':float(np.sqrt(np.mean(dgx_fp16_diff**2))),
                'one_question_paired_T':{ARMS[ai]:{LABELS[di]:float(paired(q[None,...])[0,di,ai])
                                             for di in range(2)} for ai in range(3)},
                'elapsed_seconds':float(time.time()-t),
                'scope':'one frozen question, all 68 variants per boundary, three arms',
                'is_full_30_question_result':False}
        atomic_json(pre_path,report)
        print(json.dumps({k:report[k] for k in ('status','row_id','dgx_fp16_vs_archived_windows_fp16_margin_max_abs','one_question_paired_T','elapsed_seconds')},indent=2),flush=True)
        print('=== PREFLIGHT PASSED; FULL PANEL HAS NOT RUN ===',flush=True)
        return
    margins=np.full((30,2,2,17,3,2),np.nan,dtype=np.float64)
    done=0
    sig=json.dumps(identity,sort_keys=True)
    if ckpt.exists():
        existing=np.load(ckpt,allow_pickle=False)
        if str(existing['signature'].item())!=sig:
            raise RuntimeError('Checkpoint signature mismatch')
        done=int(existing['completed_questions'].item())
        if not 0<=done<=30 or existing['margins'].shape!=margins.shape:
            raise RuntimeError('Checkpoint shape/index invalid')
        margins[:]=existing['margins']
        if not np.isfinite(margins[:done]).all() or not np.isnan(margins[done:]).all():
            raise RuntimeError('Corrupted checkpoint complete/incomplete rows')
        print(f'Resuming from {done}/30 completed questions',flush=True)
    t=time.time()
    for qi in range(done,30):
        print(f'[{qi+1:02d}/30] {row_ids[qi]}',flush=True)
        margins[qi]=run_question(qi,questions[qi],directions,control_bank,lm,m16,m32,args.batch_size)
        done=qi+1
        checkpoint_write(ckpt,margins,done,sig)
        q_diff=margins[qi,:,:,:,0,:]-archive[qi]
        print(f'  finite=YES; DGX-FP16 vs Windows-FP16 margin max_abs={np.abs(q_diff).max():.8g}; elapsed={time.time()-t:.1f}s',flush=True)
    report,paired_q,archived_paired=statistics(margins,archive,row_ids)
    report['identity']=identity
    report['status']='COMPLETE_NUMERICAL_EQUIVALENCE_MET' if report['all_four_95pct_CIs_wholly_inside_proposed_band'] else 'COMPLETE_NUMERICAL_EQUIVALENCE_NOT_MET'
    report['total_elapsed_seconds']=float(time.time()-t)
    # Write raw arrays first, then final summary. Never overwrite an existing final.
    with final_npz.open('xb') as f:
        np.savez_compressed(f,margins=margins,paired_effects=paired_q,
                            archived_windows_fp16_margins=archive,
                            archived_windows_fp16_paired_effects=archived_paired,
                            row_ids=np.array(row_ids),directions=np.array(LABELS),
                            arms=np.array(ARMS),alpha=np.array(ALPHA),boundaries=np.array(BOUNDS))
    report['raw_npz_sha256']=digest(final_npz)
    atomic_json(final_json,report)
    print(json.dumps({'status':report['status'],
          'all_four_95pct_CIs_wholly_inside_proposed_band':report['all_four_95pct_CIs_wholly_inside_proposed_band'],
          'direction_results':report['candidate_directions'],
          'saved':[str(final_npz),str(final_json)]},indent=2),flush=True)
    print('=== FULL FROZEN PRIMARY-ALPHA DGX PRECISION PANEL COMPLETE ===',flush=True)


if __name__=='__main__':
    main()
