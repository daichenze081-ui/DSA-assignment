import sys, json, math, re, html, importlib.util,reportlab
from pathlib import Path
if importlib.util.find_spec('reportlab') is None and sys.version_info[:2] == (3, 14):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tmp/pdf_deps'))
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.graphics.shapes import Drawing, Line, String, Circle, Rect
from reportlab.lib.pagesizes import A4
import fitz

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf'; OUT.mkdir(parents=True,exist_ok=True)
TMP=ROOT/'tmp/pdfs'; TMP.mkdir(parents=True,exist_ok=True)
def read(p): return json.loads((ROOT/p).read_text(encoding='utf-8'))
names=['trigram','rnn','lstm','transformer','lstm_own_frozen','lstm_glove_frozen']
results={n:read('results/'+n+'_results.json') for n in names}
hist={n:read('results/'+n+'_history.json')['history'] for n in names[1:]}
part3=read('results/part3/part3_results.json')
errors=read('results/part3/error_examples.json')
emb=read('data/embeddings/embedding_metadata.json')
labels={'trigram':'Trigram','rnn':'RNN','lstm':'LSTM','transformer':'Transformer','lstm_own_frozen':'LSTM + frozen Skip-gram','lstm_glove_frozen':'LSTM + frozen GloVe'}
navy=colors.HexColor('#17324d'); teal=colors.HexColor('#087f8c'); gray=colors.HexColor('#516172')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyX',fontName='Helvetica',fontSize=9.5,leading=13,spaceAfter=7,textColor=colors.HexColor('#243342')))
styles.add(ParagraphStyle(name='SmallX',parent=styles['BodyX'],fontSize=8,leading=10.6,spaceAfter=5))
styles.add(ParagraphStyle(name='TitleX',fontName='Helvetica-Bold',fontSize=22,leading=26,spaceAfter=12,textColor=navy))
styles.add(ParagraphStyle(name='HeadX',fontName='Helvetica-Bold',fontSize=15,leading=19,spaceAfter=10,textColor=navy))
styles.add(ParagraphStyle(name='SubX',fontName='Helvetica-Bold',fontSize=10.5,leading=14,spaceBefore=7,spaceAfter=5,textColor=teal))
styles.add(ParagraphStyle(name='CellX',parent=styles['SmallX'],fontSize=8,leading=10,spaceAfter=0))
story=[]
TABLE_CAPTIONS = iter([
    'Table 1. Data splits and token statistics.',
    'Table 2. Model architectures and gradient-trained parameter counts.',
    'Table 3. Language-model prediction and computational cost.',
    'Table 4. Generation diversity, repetition and unknown-token rates.',
    'Table 5. Part II test perplexity across optimization seeds.',
    'Table 6. Sentiment classification and selected regularization.',
    'Table 7. Descriptive accuracy by review length and UNK share.',
    'Table A1. Validation checkpoints and seed-level perplexities.',
    'Table B1. Training time per epoch and to the best checkpoint.',
    'Table B2. Autoregressive generation throughput.',
    'Table B3. Ordinary input-embedding vector norms.',
    'Table D1. Additional descriptive error groups.',
    'Table D2. Classification macro-F1 confidence intervals.',
])
def p(t,style='BodyX'): story.append(Paragraph(t,styles[style]))
def h(t): p(t,'HeadX')
def sub(t): p(t,'SubX')
def table(rows,widths=None):
    header_style=ParagraphStyle(name='White',parent=styles['CellX'],textColor=colors.white,fontName='Helvetica-Bold')
    data=[[Paragraph(html.escape(str(x)),header_style if i==0 else styles['CellX']) for x in row] for i,row in enumerate(rows)]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('TEXTCOLOR',(0,0),(-1,0),colors.white),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f4f7'),colors.white]),('LINEBELOW',(0,0),(-1,0),0.6,navy)]))
    story.extend([KeepTogether([Paragraph(next(TABLE_CAPTIONS), styles['SmallX']), t]), Spacer(1,8)])
def page(): story.append(PageBreak())
def chart(model_names, width=499,height=220):
    d=Drawing(width,height); x0=37; y0=35; w=width-52; hh=height-92
    values=[r[t]["cross_entropy"] for n in model_names for r in hist[n] for t in ["train","validation"]]
    lo=math.floor(min(values)*2)/2; hi=math.ceil(max(values)*2)/2
    max_epoch=max(len(hist[n]) for n in model_names)
    def x(e): return x0+(e-1)*w/max(1,max_epoch-1)
    def y(v): return y0+(v-lo)*hh/(hi-lo)
    palette={'lstm':teal,'rnn':colors.HexColor('#c16630'),'transformer':colors.HexColor('#6957a5'),'lstm_own_frozen':colors.HexColor('#c16630'),'lstm_glove_frozen':colors.HexColor('#6957a5')}
    for tick in [lo+i*.5 for i in range(round((hi-lo)/.5)+1)]:
        d.add(Line(x0,y(tick),width-10,y(tick),strokeColor=colors.HexColor('#dbe3e9'),strokeWidth=.5))
        d.add(String(5,y(tick)-3,f'{tick:.1f}',fontSize=8,fillColor=gray))
    for e in sorted(set([1,max_epoch]+list(range(5,max_epoch+1,5)))): d.add(String(x(e)-2,20,str(e),fontSize=8,fillColor=gray))
    d.add(String(width/2-12,5,'Epoch',fontSize=8,fillColor=gray))
    d.add(String(0,height-14,'Cross-entropy',fontName='Helvetica',fontSize=8,fillColor=gray))
    d.add(String(0,height-25,'(nats/token)',fontName='Helvetica',fontSize=8,fillColor=gray))
    # One legend above the plot, aligned to the upper right.
    lx=width-258; ly=height-44
    d.add(Rect(lx,ly,248,42,fillColor=colors.white,strokeColor=colors.HexColor('#dbe3e9'),strokeWidth=.5))
    for j,n in enumerate(model_names):
        yy=height-12-j*12
        short={'lstm_own_frozen':'Frozen Skip-gram','lstm_glove_frozen':'Frozen GloVe'}.get(n,labels[n])
        d.add(Line(lx+8,yy+2,lx+29,yy+2,strokeColor=palette[n],strokeWidth=1.7))
        d.add(String(lx+35,yy,short,fontName='Helvetica',fontSize=8,fillColor=gray))
    for j,(label,dash) in enumerate([('Training',[3,2]),('Validation',None)]):
        yy=height-15-j*16
        d.add(Line(lx+143,yy+2,lx+170,yy+2,strokeColor=navy,strokeWidth=1.4,strokeDashArray=dash))
        d.add(String(lx+176,yy,label,fontName='Helvetica',fontSize=8,fillColor=gray))
    for j,n in enumerate(model_names):
        col=palette[n]
        short={'lstm_own_frozen':'Frozen Skip-gram','lstm_glove_frozen':'Frozen GloVe'}.get(n,labels[n])
        for split in ['train','validation']:
            vals=[row[split]['cross_entropy'] for row in hist[n]]
            for k in range(len(vals)-1): d.add(Line(x(k+1),y(vals[k]),x(k+2),y(vals[k+1]),strokeColor=col,strokeWidth=1.3,strokeDashArray=[3,2] if split=='train' else None))
            for k,v in enumerate(vals): d.add(Circle(x(k+1),y(v),2,fillColor=col,strokeColor=col))
    return d

sup=ROOT/'results/supplementary'
stats=read('results/supplementary/classification.json')
grid=read('results/supplementary/trigram_grid.json')
grid_cost=read('results/supplementary/trigram_grid_cost.json')
multi=read('results/supplementary/multiseed.json')
generation=read('results/supplementary/generation.json')
diag=read('results/supplementary/embedding_diagnostics.json')
timing=read('results/supplementary/timing.json')
error_groups=read('results/supplementary/error_analysis.json')
extra=read('results/supplementary/training/transformer_budget40_results.json')
random_fixed=read('results/supplementary/training/lstm_random_frozen_results.json')
short={'trigram':'Trigram','rnn':'RNN','lstm':'LSTM','transformer':'Transformer','lstm_own_frozen':'Frozen Skip-gram','lstm_glove_frozen':'Frozen GloVe'}
def pct(x): return f'{100*x:.1f}%'
def interval(v): return f'[{100*v[0]:.1f}, {100*v[1]:.1f}]'

# Six main pages, followed by supplementary evidence and references.
p('Language Models, Embeddings,<br/>and Sentiment Representations','TitleX')
p('DSA4213 Assignment 1 | IMDb | Submission date: 16 October 2026','SmallX')
p('Name: Dai Chenze    Student ID: A0327453A','SmallX')
sub('Research question and main findings')
p('Which model predicts the next token best on a small common corpus, how do frozen embeddings change an LSTM, and do language-model representations transfer to sentiment classification? Splits are fixed and deduplicated. Hyperparameters use validation data, but supplementary experiment design was informed by original test results (p. 6).')
p(f'<b>Prediction.</b> In Part I, lower test PPL ranks LSTM ({results["lstm"]["test"]["perplexity"]:.2f}), RNN ({results["rnn"]["test"]["perplexity"]:.2f}), Transformer ({results["transformer"]["test"]["perplexity"]:.2f}), then trigram ({results["trigram"]["test"]["perplexity"]:.2f}). Grid-tuned trigram reaches {grid["test"]["perplexity"]:.2f}; Transformer cap 40 reaches {extra["test"]["perplexity"]:.2f} (Table 3). This ordering is specific to this corpus, model scale and training setup.')
p(f'<b>Embeddings.</b> Three-seed mean +/- sample SD test PPL is '+ '; '.join(f'{label}: {multi["variants"][key]["test_ppl_mean"]:.2f} +/- {multi["variants"][key]["test_ppl_std"]:.2f}' for key,label in [('random_trainable','random trainable'),('own_frozen','frozen Skip-gram'),('glove_frozen','frozen GloVe')])+f'. Random frozen reaches {random_fixed["test"]["perplexity"]:.2f} (one seed). Frozen pretrained inputs are broadly comparable to trainable inputs, without a clear overall improvement (Table 5).')
p(f'<b>Transfer.</b> TF-IDF plus logistic regression reaches {pct(stats["results"]["tfidf"]["test_accuracy"])} test accuracy, above both frozen LM representations. Transformer reaches 70.7% vs 69.1% for LSTM, but its paired accuracy difference has a 95% interval of {interval(stats["transformer_minus_lstm"]["accuracy_difference_ci95"])} percentage points and exact McNemar p={stats["transformer_minus_lstm"]["mcnemar_exact_two_sided_p"]:.3f}. This does not establish a reliable downstream ordering.')
sub('1. Data and evaluation contract')
p('IMDb is loaded as stanfordnlp/imdb [1]. Official test reviews are deduplicated first. Training reviews matching any normalized official-test text are removed; other duplicates keep their first occurrence. Keys lowercase text and collapse whitespace. With split seed 42, a stratified 6,000-review training-pool sample is divided into 5,000 training and 1,000 validation reviews; 1,000 test reviews are sampled from the separate official test pool.')
table([['Split','Reviews','Negative / positive','Tokens','UNK'],['Train','5,000','2,489 / 2,511','1,353,941','1.882%'],['Validation','1,000','498 / 502','270,132','2.813%'],['Test','1,000','498 / 502','276,278','3.304%']],[65,64,130,155,85])
p('Table 1 summarizes the split. HTML is unescaped, line breaks removed and text lowercased. A regular expression retains words with internal apostrophes, numbers and punctuation. Train-only counts define 19,996 ordinary tokens (minimum count 2), sorted by decreasing frequency with alphabetical ties, plus PAD/UNK/BOS/EOS. Unseen words map to UNK.')
p('Every model predicts from preceding context only. Cross-entropy (CE) is total negative natural-log likelihood / valid targets; PPL=exp(CE). PAD is excluded, BOS is never a target, and each review contributes one EOS: 271,132 validation and 277,278 test targets. Comparisons are conditional on this vocabulary and tokenization. The 1.35M-token training corpus limits what these models can learn.')
p('<b>Task-specific evaluation.</b> Language modeling uses CE/PPL; sentiment classification uses accuracy and macro-F1. Generated text also requires coherence assessment. These criteria answer different questions and cannot substitute for one another.','SmallX')
page()

h('2. Part I: prediction quality and cost')
table([['Model','Architecture / context','Gradient-trained parameters'],['Trigram','Count estimation; two preceding tokens','0 (count-based)'],['RNN','One tanh layer; hidden size 256','7,231,648'],['LSTM [5]','One layer; hidden size 256','7,506,592'],['Transformer [6]','2 causal layers; 4 heads; d=100; FFN=400','4,275,600']],[84,308,107])
p('Table 2 describes each architecture. Trigram training estimates count tables and probabilities from the training corpus. It learns from data, but has no neural weights optimized by gradients; the zero above excludes its learned count-based state.','SmallX')
p('Neural inputs use trainable 100-dimensional embeddings and untied output projections. BOS+tokens and tokens+EOS are sliced into aligned chunks of at most 128 positions. All review targets are retained. Recurrent states and Transformer positions reset at chunk boundaries. Transformer uses learned positions, GELU, pre-layer normalization and dropout 0.1.')
p('The original runs use Adam [7], learning rate 0.001, batch size 16, gradient clipping at 1, seed 42 and a 20-epoch cap. Three epochs without validation CE improving by more than 0.0001 from the last significant improvement trigger stopping. Every strictly lower validation CE saves a checkpoint; testing restores the absolute minimum. The 40-epoch Transformer repeats the same setup from scratch with only the cap changed.')
table([['Model / run','Test CE','Test PPL','Train s','Size MB','Score k/s']]+[[short[n],f'{results[n]["test"]["cross_entropy"]:.4f}',f'{results[n]["test"]["perplexity"]:.2f}',f'{results[n].get("training_seconds",results[n].get("count_training_seconds")):.1f}',f'{results[n].get("checkpoint_bytes",results[n].get("serialized_model_bytes"))/1e6:.2f}',f'{results[n]["inference"]["targets_per_second"]/1000:.1f}'] for n in names[:4]]+[['Trigram grid',f'{grid["test"]["cross_entropy"]:.4f}',f'{grid["test"]["perplexity"]:.2f}',f'{grid_cost["count_training_seconds"]:.1f}',f'{grid_cost["serialized_model_bytes"]/1e6:.2f}',f'{grid_cost["inference"]["targets_per_second"]/1000:.1f}'],['Transformer cap 40',f'{extra["test"]["cross_entropy"]:.4f}',f'{extra["test"]["perplexity"]:.2f}',f'{extra["training_seconds"]:.1f}',f'{extra["checkpoint_bytes"]/1e6:.2f}',f'{extra["inference"]["targets_per_second"]/1000:.1f}']],[107,62,63,72,85,110])
p(f'Table 3 notes. Train excludes validation, I/O and hyperparameter search. Trigram grid reuses the {grid_cost["count_training_seconds"]:.1f} s count fit; its size is verified after updating weights. Size is serialized bytes / 1,000,000. Score is thousands of test targets/s: first 10,000 targets, log loss included, one warm-up, median of three runs; CUDA synchronized. All timings are from the 15 October rerun with Python 3.12.13. Neural runs use RTX 3060 Laptop GPU; trigram uses CPU (Intel64 Family 6 Model 154). The same Transformer architecture scores 206.6 vs 303.9 k/s across runs (47% spread): a rough indicator of timing noise, not a formal noise estimate. This short workload and uncontrolled runtime state do not support a speed ranking.','SmallX')
sub('Interpolation and budget sensitivity')
p(f'Trigram unigram probabilities use add-0.1 smoothing. Bigram interpolation weight w2 and then trigram weight w3 give effective weights ((1-w3)(1-w2), (1-w3)w2, w3) when both contexts occur. Unseen contexts retain the lower-order distribution. A 400-pair validation grid, w2,w3 in 0,0.05,...,0.95, selects ({grid["best"]["w2"]}, {grid["best"]["w3"]}), improving on the original three-pair search. This is interpolated maximum likelihood, not Kneser-Ney smoothing.')
p(f'The extra-budget Transformer selects epoch {extra["best_epoch"]} after {extra["epochs_completed"]} epochs ({extra["stop_reason"].replace("_"," ")}), compared with epoch 19 in the original run. The {100*(1-extra["test"]["perplexity"]/results["transformer"]["test"]["perplexity"]):.2f}% test-PPL reduction leaves the original Part I ordering unchanged. Early stopping is a validation criterion, not proof of a global optimum; larger models or a different learning-rate schedule remain untested.')
p('Training time per epoch, time to the selected checkpoint and autoregressive generation speed are reported in Appendix B. Total run time includes patience epochs after the best checkpoint, so it should not be interpreted as the cost of reaching that quality.','SmallX')
page()

h('3. Part I: learning and generated text')
story.append(chart(['rnn','lstm','transformer'],height=155))
p('Figure 1. Seed-42 training (dashed) and validation (solid) cross-entropy.','SmallX')
p('RNN/LSTM validation CE rises while training CE falls after the selected epochs 6/8, consistent with overfitting. Transformer selects epoch 19 and reaches the 20-epoch cap; this alone does not establish convergence. The 40-epoch check improves PPL by only 0.40%.')
sub('Matched decoding over more prompts')
p('Generation uses the baseline seed-42 neural checkpoints and trigram interpolation weights (w2,w3)=(0.5,0.4), not the grid-selected (0.6,0.25). Six models each produce three samples for five prompts under temperature-1 full sampling and top-k=40: 180 continuations, capped at 40 new tokens and stopped at EOS. PAD/BOS are excluded; UNK is retained. Independent model/prompt/sample streams are reused across decoding policies.')
table([['Model','Full D2','Top-40 D2','Full repeat-3','Top-40 repeat-3','Top-40 UNK']]+[[short[n]]+[f'{next(r for r in generation["summary"] if r["model"]==n and r["top_k"]==k)[metric]:.3f}' for k,metric in [(None,'distinct2'),(40,'distinct2'),(None,'repeated_trigram_rate'),(40,'repeated_trigram_rate'),(40,'unk_rate')]] for n in names],[113,68,74,87,88,69])
p('In Table 4, D2 is unique bigrams / all bigrams, pooled without crossing sample boundaries. Repeat-3 is within-continuation repeated trigram occurrences / all trigram occurrences. UNK is the generated-token share. These describe output diversity, repetition and unknown tokens; none establishes semantic coherence. High trigram diversity can coexist with incoherent sentences. Length and decoding affect these measures; full results are in generation.json.','SmallX')
sub('Concrete continuations: top-k=40, sample 0')
for n in ('trigram','lstm','transformer'):
    sample=next(r for r in generation['samples'] if r['model']==n and r['top_k']==40 and r['prompt']=='the movie was' and r['sample']==0)
    p('<b>'+short[n]+':</b> '+html.escape(sample['continuation']),'SmallX')
p('In this one-prompt, sample-0 illustration, trigram fragments into punctuation/UNK; LSTM repeats the story without development; Transformer combines "great and stupid" with unclear referents. RNN starts with praise but drifts into broken syntax. Skip-gram loses its subject after "a very different, but"; GloVe shifts from a TV show to an unclear viewing account. These are readings of examples, not model-wide judgments (Appendix C).','SmallX')
p('Only 15 continuations per model/decoder support Table 4. A repeat-3 rate of 0.005 is about three repeated occurrences among roughly 600 trigrams. Small D2/repeat-3 differences cannot rank quality; larger D2 differences still describe lexical diversity, not coherence.','SmallX')
page()

h('4. Part II: embeddings and seed variability')
p('The Part I LSTM is reused as the random-trainable baseline. Own vectors use Skip-gram negative sampling [3,4]: dimension 100, window 5, five negatives, min count 2, subsampling 0.001, five epochs, one worker, seed 42. Public vectors are GloVe 6B 100d [2]. Both pretrained tables are entirely frozen, including missing and special rows; remaining LSTM parameters train from scratch. Trainable parameters decrease from 7,506,592 to 5,506,592.')
p('All variants use the same splits, architecture, Adam settings and stopping policy. Seeds 42/43/44 vary LM initialization and batch order. Pretrained vectors and Normal(0,1) missing rows remain fixed at seed 42, with zero PAD. Mean and sample standard deviation below therefore measure optimization variability conditional on these embeddings, not variability from retraining Skip-gram or resampling data.')
table([['LSTM input','Seed 42 PPL','Seed 43 PPL','Seed 44 PPL','Mean +/- SD','Train to best s (seed 42)']]+[[{'random_trainable':'Random trainable','own_frozen':'Frozen Skip-gram','glove_frozen':'Frozen GloVe'}[n]]+[f'{r["test_perplexity"]:.2f}' for r in d['runs']]+[f'{d["test_ppl_mean"]:.2f} +/- {d["test_ppl_std"]:.2f}', f'{timing[{"random_trainable":"lstm","own_frozen":"lstm_own_frozen","glove_frozen":"lstm_glove_frozen"}[n]]["train_seconds_to_best"]:.1f}'] for n,d in multi['variants'].items()],[106,64,64,64,102,99])
p('Time is seed-42 training through the best checkpoint, excluding validation/pretraining. The LSTM architecture is identical: 21.5/28.2/37.8 s per epoch differ by 75%, indicating uncontrolled runtime variation, not an identified cost of freezing. Times cannot rank embedding efficiency. Three seeds and one split do not establish general superiority.','SmallX')
p(f'A seed-42 random-frozen control gives test PPL {random_fixed["test"]["perplexity"]:.2f}, versus {results["lstm"]["test"]["perplexity"]:.2f} with identical initialization but trainable inputs: freezing harms prediction here. Both pretrained-frozen tables improve on random-frozen at seed 42, recovering much of this loss without input trainability. Domain, objective and scale still differ; this control is one seed.')
story.append(chart(['lstm','lstm_own_frozen','lstm_glove_frozen'],height=165))
p('Figure 2. Seed-42 learning curves: best epochs 8/11/10, stopping at 11/14/13. Freezing delays the optimum here but does not prevent overfitting. Table 5 supplies the evidence about seed variability.','SmallX')
sub('Coverage, scale and cost')
p(f'Skip-gram covers all 19,996 ordinary LM words and 100% of their training occurrences. GloVe covers 19,042/19,996 words (95.23%) and {100*diag["glove_lm_token_coverage"]:.2f}% of occurrences restricted to ordinary LM words. It leaves 954 random ordinary rows. Relative to all original training tokens (including out-of-vocabulary words), coverage is 98.12%/96.53%. These denominators answer different questions.')
p('Mean ordinary-row L2 norms: '+', '.join(f'{label} {diag["norms"][key]["mean_ordinary_l2_norm"]:.2f}' for key,label in [('random_initial','random initial'),('lstm_own_frozen','Skip-gram'),('lstm_glove_frozen','GloVe')])+'. Unequal scale is a remaining confound; no normalized-table ablation was performed. Differences cannot be attributed solely to pretraining quality.')
p('Skip-gram uses only the same 1.35M-token training corpus: it adds no data yet roughly matches trainable inputs. GloVe adds 6B external Wikipedia/Gigaword tokens [2] without winning here; domain mismatch with reviews, 954 random ordinary-word rows and unequal norms are possible explanations, none isolated by this experiment.','SmallX')
p(f'Skip-gram pretraining costs {emb["own"]["training_seconds"]:.2f} s separately. A local GloVe parsing pass costs {diag["glove_parse_seconds_measured_now"]:.2f} s; historical download time was not recorded, and public pretraining cost is excluded. Per-epoch LM costs are in Appendix B.','SmallX')
page()

h('5. Part III: sentiment representations')
p('The baseline Part I checkpoints are frozen. Contextual states before the vocabulary projection are averaged over ordinary and UNK token positions, excluding PAD/BOS/EOS. Chunk states are weighted by token count, retaining the causal/reset policy. LSTM and Transformer features have 256 and 100 dimensions.')
p('Logistic regression (L-BFGS, max 2,000 iterations) selects C in {0.001,0.01,0.1,1,10} by validation macro-F1. Dense-feature scaling and the 100d PCA control are train-fit. TF-IDF uses the same tokenizer, 1-2 grams, min_df=2, at most 50,000 features, sublinear TF and L2 normalization; vocabulary/IDF are train-fit. No external vectors enter this baseline.')
table([['Representation','Selected C','Val. macro-F1','Test accuracy','95% accuracy CI','Test macro-F1']]+[[label,f'{(stats["results"][n] if "best_C" in stats["results"][n] else part3["results"][n])["best_C"]:g}',f'{max(r["validation_macro_f1"] for r in (stats["results"][n].get("validation_search") or part3["results"][n]["validation_search"])):.4f}',pct(stats['results'][n]['test_accuracy']),interval(stats['results'][n]['accuracy_ci95']),f'{stats["results"][n]["test_macro_f1"]:.4f}'] for n,label in [('lstm','LSTM mean, 256d'),('transformer','Transformer mean, 100d'),('lstm_pca100','LSTM + PCA, 100d'),('tfidf','TF-IDF, 50,000d')]],[125,54,76,70,98,76])
p('Table 6 intervals use 10,000 paired test-review bootstrap resamples (seed 4213), with fixed models; they exclude retraining variability. Majority accuracy is 50.2%. LSTM (C=0.1) and Transformer (C=0.01) are now interior selections. PCA (C=0.001) and TF-IDF (C=10) remain at grid boundaries. Values beyond this grid remain untested, so optimal regularization is not established.','SmallX')
p(f'Transformer makes 293 errors vs 309 for LSTM. Their discordant counts are 111 (LSTM only correct) and 127 (Transformer only correct). Exact two-sided McNemar p={stats["transformer_minus_lstm"]["mcnemar_exact_two_sided_p"]:.3f}; the paired macro-F1 difference interval is [{stats["transformer_minus_lstm"]["macro_f1_difference_ci95"][0]:.4f}, {stats["transformer_minus_lstm"]["macro_f1_difference_ci95"][1]:.4f}]. Both difference intervals include zero. Validation favors LSTM (0.7709 vs 0.7537), so the test ordering is not a stable architecture ranking.')
p('Table 6 shows a stronger TF-IDF baseline. PCA does not close the gap (67.0% accuracy); it controls dimension, not architecture. Validation-to-test macro-F1 drops are 0.0799 for LSTM, 0.0469 for Transformer and 0.0270 for TF-IDF. The TF-IDF drop shows that LM exposure cannot be the sole explanation; sampling differences and validation-based selection may contribute.')
p('My reading of Appendix D: #922 and #605 mix praise with rejection of execution, while #254 uses positive words in a negative judgment and #447 explicitly praises a film classified negative. The full #965 review mixes pacing criticism with a positive ending; a truncated excerpt can hide that conclusion. #987 appears positive despite its negative label, suggesting a possible label inconsistency, not a verified error in annotation.','SmallX')
sub('Error groups and unverified split hypothesis')
table([['Review group','n','LSTM accuracy','Transformer accuracy']]+[[k,error_groups['groups']['lstm'][k]['n'],pct(error_groups['groups']['lstm'][k]['accuracy']),pct(error_groups['groups']['transformer'][k]['accuracy'])] for k in ['length<=128','length>256','UNK>2%']],[184,55,130,130])
p('Table 7 groups overlap. The short-review group has only n=158: the 3.16-point Transformer advantage is five reviews; paired exact McNemar p=0.533 (18 vs 23 discordant cases), compatible with sampling noise. Appendix D contains randomly sampled errors.','SmallX')
p('Maas et al. [8] constructed official train/test sets with disjoint movies. Our train/validation subsets both come from the official train pool, so they may share movies. HF plain_text exposes text and label, not movie IDs [1]; overlap is unverified here. Both validation and test reviews were unseen by the LM: exposure to training reviews alone cannot explain their contrast. Representation overfitting could contribute to the LM-specific excess drop over TF-IDF, but this is an untested hypothesis, not a causal decomposition.','SmallX')
page()

h('6. Interpretation and reproducibility')
sub('What the comparisons support')
p('<b>Prediction quality is conditional on the experiment.</b> LSTM achieves lower CE/PPL than Transformer on this small corpus at the tested model sizes and training settings. The trigram grid and extra-budget Transformer probe two comparison weaknesses. Possible explanations are not isolated: Transformer/LSTM have 4.28M/7.51M parameters, including 2.02M/5.14M in their output layers. Both use chunks of at most 128 positions, versus two tokens for trigram. Completed epochs and training times also differ (Table 3). These factors may contribute; this is not a universal architecture ranking.')
best_variant=min(multi['variants'],key=lambda n:multi['variants'][n]['test_ppl_mean'])
best_label={'random_trainable':'random trainable embeddings','own_frozen':'frozen Skip-gram','glove_frozen':'frozen GloVe'}[best_variant]
p(f'<b>Frozen pretraining roughly matches trainable inputs.</b> The lowest three-seed mean test PPL belongs to {best_label}. Three seeds are a small estimate of optimization variability, with fixed embeddings. The random-frozen control helps separate freezing from vector source; unmatched norms, external GloVe data and coverage still limit attribution. A tiny single-seed improvement should not be used as evidence of general superiority.')
p('<b>Language modeling and sentiment need different evidence.</b> Table 6 shows a stronger supervised lexical baseline than either frozen LM feature set. LSTM has lower LM perplexity (170 vs 205). Transformer is nominally higher on downstream test accuracy, but its paired difference CI includes zero; validation favors LSTM. There is no evidence establishing either agreement or reversal of the two rankings. Random error examples in Appendix D illustrate failures but do not identify their internal causes.')
sub('Remaining limits')
p('Training uses only 1.35M tokens and one split. Multi-seed runs hold pretrained tables fixed; the architecture comparison, extra-budget run and random-frozen control are single-seed. Generation uses five prompts, three samples each and no human fluency annotation. Diversity does not measure meaning. Exact deduplication misses near-duplicates. Test bootstrap intervals omit training variability; error groups overlap. Supplementary experiments were designed after inspecting original test results. Validation-based hyperparameter selection does not undo that influence on the research process: the test set is no longer untouched. Supplementary findings are exploratory evidence; confirmation requires fresh held-out data.')
sub('Reproduction package')
p('Assignment.ipynb is the complete execution path: restart the kernel and run all cells after preparing dependencies and GloVe. Original and extended experiments, result checks and provenance are integrated in topic order, with no automatic reuse of old training JSON. Section notes map inputs to output files. scripts/build_report.py only formats the PDF. The complete notebook was rerun on 15 October 2026, 00:16-01:43 Singapore time, and reached the final checks. On 16 October, only classifiers and their statistics were rerun from saved features with the expanded C grid; run_provenance.json records this partial update separately. LM results, generation and timings remain from the full run.')
p('The experiments used Python 3.12.13 and package versions recorded in requirements.txt and results/supplementary/environment.json. Neural experiments use PyTorch 2.14.0+cu130 on an RTX 3060 Laptop GPU; CPU execution is supported but slower and timings will differ. IMDb and GloVe 6B 100d are obtained from the cited sources; the expected vector path is data/glove.6B/glove.6B.100d.txt. Split and vocabulary hashes are retained for checking reproduction. Exact numerical identity across devices is not guaranteed.')
sub('Sources, attribution and tool use')
p('The implementation uses PyTorch, gensim and scikit-learn. Model, algorithm and dataset references [1-8] appear after the appendices. No course notebook was adapted. OpenAI Codex assisted with code and PDF preparation. I am responsible for the experimental design and conclusions and have checked them.')
p('The code archive includes the notebook, PDF builder, report, README, requirements, numeric results and saved test predictions. It excludes raw review splits, token-ID files, GloVe vectors, model checkpoints and classifier feature/model files. GloVe 6B 100d must be downloaded; the notebook regenerates other excluded artifacts. The full local working folder also retains these artifacts. README documents the same packaging distinction.','SmallX')
page()

h('Appendix A. Additional training evidence')
table([['Variant / seed','Best epoch','Test PPL','Validation PPL']]+[[n.replace('_',' ')+' / '+str(r['seed']),r['best_epoch'],f'{r["test_perplexity"]:.3f}',f'{r["validation_perplexity"]:.3f}'] for n,d in multi['variants'].items() for r in d['runs']],[205,80,107,107])
sub('Extra-budget Transformer learning curve')
hist['transformer']=read('results/supplementary/training/transformer_budget40_history.json')['history']
story.append(chart(['transformer'],height=200))
p(f'Figure A1. Extra-budget Transformer training and validation CE. Table A1 reports seed-level results. The same-seed fresh run executes {extra["epochs_completed"]} epochs; selected epoch {extra["best_epoch"]}, stop reason {extra["stop_reason"]}. Test CE {extra["test"]["cross_entropy"]:.6f}, PPL {extra["test"]["perplexity"]:.4f}. The optimizer and learning rate are unchanged. Its history is saved separately from the original 20-epoch experiment.','SmallX')
p(f'Random-frozen control: best epoch {random_fixed["best_epoch"]}, {random_fixed["epochs_completed"]} epochs executed, test PPL {random_fixed["test"]["perplexity"]:.4f}. The complete input table is frozen, including special rows.','SmallX')
page()
h('Appendix B. Timing and embedding diagnostics')
table([['Model / embedding (seed 42)','Train s / epoch','Train s to best','Train + validation s to best']]+[[short[n],f'{d["mean_train_seconds_per_epoch"]:.2f}',f'{d["train_seconds_to_best"]:.2f}',f'{d["train_and_validation_seconds_to_best"]:.2f}'] for n,d in timing.items()],[155,100,110,134])
p('Table B1 measures time to best across epochs 1 through the selected epoch, and excludes checkpoint I/O and preprocessing. Per-epoch means cover all executed epochs. These timings come from the complete 15 October rerun. Sequential measurements can reflect changing runtime conditions; they do not isolate architectural efficiency.','SmallX')
table([['Model','Full sample tokens/s','Top-40 tokens/s','Full generated tokens','Top-40 generated tokens']]+[[short[n]]+[f'{next(r for r in generation["summary"] if r["model"]==n and r["top_k"]==k)[metric]:.1f}' for k,metric in [(None,'tokens_per_second'),(40,'tokens_per_second'),(None,'tokens'),(40,'tokens')]] for n in names],[125,102,88,94,90])
p('Table B2 measures batch-one autoregressive generation: five prompts x three samples; warm-up excluded; CUDA synchronized; median of three full-workload timings. Numerator counts emitted non-EOS tokens, and elapsed time includes prompt setup/tokenization. EOS can shorten outputs. Trigram is CPU; neural models GPU. Recurrent state is cached; Transformer recomputes context without a KV cache. Throughput characterizes these implementations and sampling settings.','SmallX')
table([['Embedding table','Mean ordinary L2','Std ordinary L2']]+[[{'random_initial':'Random initialization','lstm':'LSTM: random trainable','lstm_own_frozen':'LSTM: frozen Skip-gram','lstm_glove_frozen':'LSTM: frozen GloVe'}[n],f'{d["mean_ordinary_l2_norm"]:.3f}',f'{d["std_ordinary_l2_norm"]:.3f}'] for n,d in diag['norms'].items()],[265,117,117])
p(f'Table B3 summarizes vector norms. GloVe covers {diag["glove_covered_lm_token_occurrences"]:,} of {diag["ordinary_lm_token_occurrences"]:,} ordinary-LM-word occurrences in training. PAD and special-row norms are recorded separately in embedding_diagnostics.json. No rescaling was applied. The local parsing cost is a current measurement, not the historical download or pretraining cost.','SmallX')
page()
h('Appendix C. Matched generation examples')
p('The five prompts are: '+ '; '.join(html.escape(x) for x in generation['prompts'])+'. The excerpts below use the movie was, sample 0, selected by index for every model/policy. All 180 continuations and token IDs are saved in generation.json. Trigram generation uses (w2,w3)=(0.5,0.4); Table 4 reports diversity and Table B2 reports speed.','SmallX')
for k in (None,40):
    sub('Temperature 1; '+('full distribution' if k is None else 'top-k=40'))
    for n in names:
        r=next(r for r in generation['samples'] if r['model']==n and r['top_k']==k and r['prompt']=='the movie was' and r['sample']==0)
        p('<b>'+short[n]+f' (seed {r["seed"]}):</b> '+html.escape(r['continuation']),'SmallX')
page()
h('Appendix D. Error analysis and paired uncertainty')
table([['Group','n','LSTM acc.','Transformer acc.','TF-IDF acc.']]+[[k,d['n']]+[pct(error_groups['groups'][n][k]['accuracy']) for n in ['lstm','transformer','tfidf']] for k,d in error_groups['groups']['lstm'].items() if not k.startswith('negation_')],[179,50,90,95,85])
p('Table D1 uses token-ID length and per-review UNK share; groups overlap. The broad negation proxy is omitted. For n=158 short reviews, 18 cases favor LSTM and 23 favor Transformer; the 3.16-point difference has exact paired McNemar p=0.533. This post-hoc subgroup check is exploratory.','SmallX')
p('Table D2 reports macro-F1 uncertainty. The following page examines four random errors and two high-confidence LSTM errors; 0 denotes negative and 1 positive.','SmallX')
table([['Representation','Macro-F1','95% bootstrap interval']]+[[{'tfidf':'TF-IDF, 50,000d','lstm_pca100':'LSTM + PCA, 100d','lstm':'LSTM mean, 256d','transformer':'Transformer mean, 100d'}[n],f'{d["test_macro_f1"]:.4f}',f'[{d["macro_f1_ci95"][0]:.4f}, {d["macro_f1_ci95"][1]:.4f}]'] for n,d in stats['results'].items()],[189,105,205])
page()
h('Appendix D (continued). Reading individual errors')
p('The first four cases are the first two saved random errors per LM (seed 4213); the last two are the most confident LSTM errors in each direction. Selected passages use [...] for omissions; full reviews remain in the saved error JSON. Each interpretation below is my reading, not a demonstrated model mechanism or an estimate of error-type prevalence.','SmallX')
readings = {
    965: ('"Begotten" is one of the most visually dazzling and mystifying films ever known to man. [...] It\'s just that everything happens to such a slow pace. [...] Still,it\'s....I don\'t know....Glorious...',
          'The opening excerpt emphasizes praise, but truncating the review hides its final judgment. The full text also criticizes pacing before ending positively; negative plot vocabulary and reservations may distract from that verdict, but this mechanism is untested.'),
    922: ('I liked the main characters and the overall story but some scenes are pretty sloppy and confusing. [...] Sound design was a bit poor. I think less would have been best.',
          'Affection for the characters and story coexists with criticism of execution and sound. I read this as mixed sentiment whose overall negative judgment is missed.'),
    987: ('I really liked this quirky movie. [...] It has a realistic edge, with a captivating story line. The main title sequence alone makes this movie fun to watch.',
          'Even the full review appears positive, despite the recorded negative label. A label inconsistency is plausible but unverified; I retain the supplied label and do not count this as confirmed annotation noise.'),
    254: ('A lot of actors have a multitude of good movie roles in their soul. [...] Makes you forget what a terrible movie you\'re watching. [...] Two stars.',
          'The positive phrase about movie roles frames a sarcastic, negative review; the explicit verdict is poor. Positive vocabulary alone would misrepresent its sentiment.'),
    605: ('Great premise, poor execution. Cast of great actors is watered down into a poorly written, poorly directed, poorly edited, waste of film.',
          'Praise for the premise and cast is outweighed by direct criticism of execution. This is a clear negative verdict despite positive words; high classifier confidence does not make it reliable.'),
    447: ('This movie is a masterpiece. Go see it. There\'s nothing more to say.',
          'The recommendation and repeated praise in the full text are unambiguously positive to me. Words such as sorrow occur within praise, so isolated negative cues would lose the surrounding meaning; that explanation remains a hypothesis.'),
}
for model, index in [('lstm',965),('lstm',922),('transformer',987),('transformer',254),('lstm',605),('lstm',447)]:
    pool = errors[model] if index in (605,447) else error_groups['random_errors'][model]
    e = next(row for row in pool if row['test_index']==index)
    excerpt, interpretation = readings[index]
    confidence = f'; predicted-class probability {e["confidence"]:.4f}' if 'confidence' in e else ''
    p(f'<b>{short[model]} #{index}: {e["true_label"]} -&gt; {e["predicted_label"]}{confidence}.</b> '+html.escape(excerpt),'SmallX')
    p('<i>My reading:</i> '+html.escape(interpretation),'SmallX')
page()
h('References and source notes')
refs=[
('[1] Stanford NLP. IMDb dataset card. Dataset identifier: stanfordnlp/imdb.','https://huggingface.co/datasets/stanfordnlp/imdb'),
('[2] Pennington, J., Socher, R., and Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. EMNLP. Public vectors: GloVe 6B, 100d.','https://nlp.stanford.edu/pubs/glove.pdf'),
('[3] Mikolov, T., Chen, K., Corrado, G., and Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space.','https://arxiv.org/abs/1301.3781'),
('[4] Mikolov, T., Sutskever, I., Chen, K., Corrado, G., and Dean, J. (2013). Distributed Representations of Words and Phrases and their Compositionality. NeurIPS.','https://arxiv.org/abs/1310.4546'),
('[5] Hochreiter, S., and Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation 9(8), 1735-1780.','https://www.bioinf.jku.at/publications/older/2604.pdf'),
('[6] Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.','https://arxiv.org/abs/1706.03762'),
('[7] Kingma, D. P., and Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR.','https://arxiv.org/abs/1412.6980'),
('[8] Maas, A. L., et al. (2011). Learning Word Vectors for Sentiment Analysis. ACL, pp. 142-150. Dataset split construction: Sections 4.3.1-4.3.2.','https://aclanthology.org/P11-1015/')]
for text,url in refs:
    p(html.escape(text)+f'<br/><link href="{url}" color="#087f8c">{url}</link>')
p('GloVe data access: <link href="https://nlp.stanford.edu/projects/glove/" color="#087f8c">https://nlp.stanford.edu/projects/glove/</link>. The vectors are glove.6B.100d.txt from glove.6B.zip. IMDb split construction and HF fields were verified on 15 October 2026.','SmallX')
sub('Artifact provenance')
p('Numerical tables are read directly from saved experiment JSON, feature-based classifier predictions and training histories. The report builder does not train models. Supplementary analysis keeps the original split, labels and vocabulary. The submission code package includes a file manifest; raw external corpora and public embedding files are obtained from their original sources.','SmallX')

def footer(c,doc):
    c.setStrokeColor(colors.HexColor('#dbe3e9')); c.line(48,43,A4[0]-48,43)
    c.setFont('Helvetica',7.5); c.setFillColor(gray)
    c.drawString(48,29,'DSA4213 | IMDb language models and sentiment representations')
    c.drawRightString(A4[0]-48,29,str(doc.page))
path=OUT/'DSA4213_Assignment1.pdf'
doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=40,bottomMargin=55,title='DSA4213 Assignment 1 - Experimental Report')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
pdf=fitz.open(path)
for i,pg in enumerate(pdf):
    print(i+1,pg.get_text().splitlines()[0],len(pg.get_text()))
    pg.get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(TMP/f'revised-page-{i+1:02}.png')
assert 'Appendix A.' in pdf[6].get_text(), 'Main report must fit exactly six pages'
assert all(len(pg.get_text())>200 for pg in pdf), 'Unexpected near-empty page'
print('PDF',path,'PAGES',len(pdf))

