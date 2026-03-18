"""
ArvyaX ML Pipeline — Optimized Final Version
987 training samples, sample-weighted ensemble, full decision engine
"""
import pandas as pd, numpy as np, warnings, os, pickle
warnings.filterwarnings('ignore')
np.random.seed(42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier, XGBRegressor

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE,'outputs'), exist_ok=True)
os.makedirs(os.path.join(BASE,'models'),  exist_ok=True)

POSITIVE_WORDS=['calm','peaceful','settled','lighter','clear','focused','grounded','better',
 'organized','ease','quiet','softened','breathing','slowed','prioritize','ready',
 'sharp','plan','concentrate','able','think','straight','good','more settled','less tense']
NEGATIVE_WORDS=['racing','flooded','overwhelmed','exhausted','heavy','anxious','restless',
 'distracted','jumpy','drained','pressure','scattered','unsettled','fidgety','buzz',
 'overloaded','piled','itchy','behind','crowded','wandered','unable to','hard to focus',
 'mind racing','all over','too much','mentally flooded','still carrying']
MIXED_WORDS=['but','however','though','yet','still','between','split','surface',
 'part of me','two moods','not completely','better and not']
UNCERTAINTY_W=['idk','not sure','can\'t tell','maybe','unclear','couldn\'t tell',
 'i guess','kinda','somehow','not fully','not completely','not sure why']
SHORT_P=['ok','fine','ok session','still off','mind racing','bit restless','more clear today',
 'hard to focus','not much change','actually helped','felt better','tired but okay',
 'a little lighter','mixed honestly','kinda calm','okay session','it was fine',
 'could focus','breathing slowed','helped me plan','felt lighter','felt good',
 'still heavy','felt heavy','back to normal','got distracted','mind was all over',
 'still anxious','some peace']

def text_features(text):
    if pd.isna(text) or str(text).strip()=='': text='unknown'
    tl=str(text).lower(); words=tl.split()
    pos=sum(1 for w in POSITIVE_WORDS if w in tl)
    neg=sum(1 for w in NEGATIVE_WORDS if w in tl)
    mix=sum(1 for w in MIXED_WORDS    if w in tl)
    unc=sum(1 for w in UNCERTAINTY_W  if w in tl)
    wc=len(words)
    has_tmpl=int(any(p in tl for p in ['at one point','after a few minutes','by the end',
        'gradually','strangely','for some reason','during the session','at first','i noticed']))
    return {
        'pos_score':pos,'neg_score':neg,'mixed_score':mix,'uncertainty_score':unc,
        'sentiment':(pos-neg)/(pos+neg+1),'is_short':int(wc<=5),
        'is_vague_phrase':int(any(p in tl for p in SHORT_P)),
        'has_contrast':int('but' in tl or 'however' in tl or 'yet' in tl),
        'has_body_cue':int(any(w in tl for w in ['breathing','shoulder','chest','body','tense'])),
        'has_task_cue':int(any(w in tl for w in ['work','emails','tasks','plan','focus','prioritize'])),
        'has_restart':int('restart' in tl or 'wandered' in tl),
        'has_template':has_tmpl,'word_count':wc,'char_count':len(str(text)),
        'avg_word_len':np.mean([len(w) for w in words]) if words else 0,
    }

def meta_features(row):
    def s(k,d): v=row.get(k,d); return float(v) if pd.notna(v) and str(v).strip()!='' else float(d)
    sleep=s('sleep_hours',6); energy=s('energy_level',3); stress=s('stress_level',3); dur=s('duration_min',15)
    tm={'morning':0,'early_morning':0,'afternoon':1,'evening':2,'night':3}
    am={'forest':0,'ocean':1,'rain':2,'mountain':3,'cafe':4}
    mm={'very_low':-2,'low':-1,'neutral':0,'mixed':0,'focused':1,'calm':1,'restless':-1,'overwhelmed':-2,'':0}
    fm={'calm_face':2,'happy_face':2,'neutral_face':0,'tired_face':-1,'tense_face':-2,'none':0,'':0}
    qm={'clear':2,'conflicted':0,'vague':-1,'':0}
    tod=tm.get(str(row.get('time_of_day','')).lower(),1)
    return {'sleep_hours':sleep,'sleep_quality':(sleep-3)/6,'energy_level':energy,'stress_level':stress,
        'duration_min':dur,'energy_stress_ratio':energy/(stress+0.1),
        'low_sleep':int(sleep<5),'high_stress':int(stress>=4),'low_energy':int(energy<=2),
        'time_of_day_enc':tod,'is_night':int(tod==3),'is_morning':int(tod==0),
        'ambience_enc':am.get(str(row.get('ambience_type','')).lower(),0),
        'prev_mood_enc':mm.get(str(row.get('previous_day_mood','')).lower(),0),
        'face_emotion_enc':fm.get(str(row.get('face_emotion_hint','')).lower(),0),
        'reflection_quality_enc':qm.get(str(row.get('reflection_quality','')).lower(),0)}

def build_features(df, tfidf=None, fit=False):
    tf=pd.DataFrame([text_features(t) for t in df['journal_text'].fillna('')])
    mf=pd.DataFrame([meta_features(r) for _,r in df.iterrows()])
    txts=df['journal_text'].fillna('').tolist()
    if fit:
        tfidf=TfidfVectorizer(max_features=120,ngram_range=(1,2),stop_words='english',min_df=2,max_df=0.9)
        tmat=tfidf.fit_transform(txts)
    else:
        tmat=tfidf.transform(txts)
    tdf=pd.DataFrame(tmat.toarray(),columns=[f'tfidf_{i}' for i in range(tmat.shape[1])])
    return pd.concat([tf.reset_index(drop=True),mf.reset_index(drop=True),tdf.reset_index(drop=True)],axis=1),tfidf

WHAT_RULES={'calm':{'default':'light_planning','morning':'deep_work','night':'journaling'},
 'focused':{'default':'deep_work','night':'light_planning','tired':'rest'},
 'neutral':{'default':'light_planning','morning':'deep_work','night':'rest'},
 'restless':{'default':'box_breathing','high_stress':'grounding','night':'sound_therapy'},
 'mixed':{'default':'journaling','morning':'box_breathing','night':'rest'},
 'overwhelmed':{'default':'grounding','night':'rest','low_energy':'rest','morning':'box_breathing'}}
WHEN_RULES={'calm':{'morning':'now','afternoon':'now','evening':'later_today','night':'tomorrow_morning'},
 'focused':{'morning':'now','afternoon':'now','evening':'within_15_min','night':'now'},
 'neutral':{'morning':'within_15_min','afternoon':'later_today','evening':'later_today','night':'tomorrow_morning'},
 'restless':{'morning':'now','afternoon':'now','evening':'within_15_min','night':'tonight'},
 'mixed':{'morning':'within_15_min','afternoon':'later_today','evening':'tonight','night':'tomorrow_morning'},
 'overwhelmed':{'morning':'now','afternoon':'now','evening':'within_15_min','night':'tonight'}}
MSGS={'calm':"You seem calm and grounded. This is a good window to use your clarity intentionally.",
 'focused':"Your mind is sharp and ready. Use this momentum — start with your most important task.",
 'neutral':"You're in a steady state. A little structure can help you make the most of this window.",
 'restless':"There's restless energy in your system. Let's slow things down before diving in.",
 'mixed':"You're carrying two states at once — some calm, some tension. Let's process it first.",
 'overwhelmed':"You seem overwhelmed. Before anything else, let's create a little space to breathe."}
ACTS={'box_breathing':"Try 4 rounds of box breathing (4s inhale → hold → exhale → hold).",
 'journaling':"Write freely for 5 minutes. Let what's inside come out on the page.",
 'grounding':"5-4-3-2-1: 5 things you see, 4 feel, 3 hear, 2 smell, 1 taste.",
 'deep_work':"Block 45–90 min for deep work. Turn off notifications. You're ready.",
 'rest':"Your system needs recovery. Take a proper break — nap, walk, or quiet time.",
 'movement':"A short walk (10 min) will help discharge restless energy.",
 'sound_therapy':"Try calming ambient sounds for 10–15 minutes before sleep.",
 'light_planning':"Spend 10 minutes writing your 3 priorities. Keep it light and clear.",
 'yoga':"10 minutes of gentle yoga will bridge tension and calm.",
 'pause':"Take a deliberate 5-minute pause before your next activity."}

def decide(state,row):
    def s(k,d): v=row.get(k,d); return float(v) if pd.notna(v) and str(v).strip()!='' else float(d)
    stress=s('stress_level',3); energy=s('energy_level',3)
    tod=str(row.get('time_of_day','')).lower()
    tod_b='morning' if 'morning' in tod else (tod if tod in ['afternoon','evening','night'] else 'afternoon')
    rules=WHAT_RULES.get(state,{'default':'journaling'})
    if stress>=4 and 'high_stress' in rules: what=rules['high_stress']
    elif energy<=2 and 'low_energy' in rules: what=rules['low_energy']
    elif tod_b in rules: what=rules[tod_b]
    else: what=rules['default']
    wr=WHEN_RULES.get(state,{'morning':'now'}); when=wr.get(tod_b,'within_15_min')
    if stress>=5: when='now'
    if tod_b=='night' and state in ['calm','neutral']: when='tomorrow_morning'
    return what,when,(MSGS.get(state,'')+' '+ACTS.get(what,'')).strip()

def compute_unc(text,qual,proba):
    max_p=float(max(proba)); margin=max_p-sorted(proba)[-2]; tl=str(text).lower()
    unc=sum(1 for w in UNCERTAINTY_W if w in tl); is_short=int(len(tl.split())<=5)
    qa={'clear':0.0,'conflicted':-0.10,'vague':-0.15}.get(str(qual).lower(),0.0)
    conf=float(np.clip(max_p*0.6+margin*0.3-unc*0.03-is_short*0.05+qa,0.10,0.97))
    flag=int(conf<0.55 or is_short or str(qual).lower()=='conflicted' or unc>=2)
    return round(conf,3),flag

def run_pipeline(train_path=None,test_path=None,out_path=None):
    train_path=train_path or os.path.join(BASE,'data/train_weighted.csv')
    test_path =test_path  or os.path.join(BASE,'data/test.csv')
    out_path  =out_path   or os.path.join(BASE,'outputs/predictions.csv')
    print("="*60+"\n  ArvyaX ML Pipeline — Optimized Final\n"+"="*60)
    df_tr=pd.read_csv(train_path); df_te=pd.read_csv(test_path)
    df_tr['intensity']=pd.to_numeric(df_tr['intensity'],errors='coerce').fillna(3).clip(1,5).astype(int)
    sw=df_tr['quality_weight'].values if 'quality_weight' in df_tr.columns else None
    print(f"\nTrain:{df_tr.shape} | Test:{df_te.shape}")
    print(df_tr['emotional_state'].value_counts().to_string())
    print("\n[Features] Building...")
    X_tr,tfidf=build_features(df_tr,fit=True)
    X_te,_    =build_features(df_te,tfidf=tfidf,fit=False)
    imp=SimpleImputer(strategy='mean')
    X_tr=pd.DataFrame(imp.fit_transform(X_tr),columns=X_tr.columns)
    X_te=pd.DataFrame(imp.transform(X_te),    columns=X_te.columns)
    print(f"Feature matrix: {X_tr.shape}")
    le=LabelEncoder(); y=le.fit_transform(df_tr['emotional_state']); yi=df_tr['intensity']
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    print("\n[Training] Ensemble classifiers...")
    xgb=XGBClassifier(n_estimators=400,max_depth=5,learning_rate=0.07,subsample=0.8,
        colsample_bytree=0.8,use_label_encoder=False,eval_metric='mlogloss',
        min_child_weight=2,gamma=0.1,random_state=42,n_jobs=-1)
    rf =RandomForestClassifier(n_estimators=400,max_depth=12,min_samples_leaf=2,random_state=42,n_jobs=-1)
    lr =LogisticRegression(C=0.5,max_iter=2000,random_state=42)
    xgb.fit(X_tr,y,sample_weight=sw); rf.fit(X_tr,y,sample_weight=sw); lr.fit(X_tr,y)
    xgb_cv=cross_val_score(xgb,X_tr,y,cv=cv,scoring='accuracy')
    rf_cv =cross_val_score(rf, X_tr,y,cv=cv,scoring='accuracy')
    lr_cv =cross_val_score(lr, X_tr,y,cv=cv,scoring='accuracy')
    print(f"  XGBoost  CV Acc: {xgb_cv.mean():.3f} ± {xgb_cv.std():.3f}")
    print(f"  RF       CV Acc: {rf_cv.mean():.3f}  ± {rf_cv.std():.3f}")
    print(f"  LogReg   CV Acc: {lr_cv.mean():.3f}  ± {lr_cv.std():.3f}")
    ens=xgb.predict_proba(X_te)*0.50+rf.predict_proba(X_te)*0.35+lr.predict_proba(X_te)*0.15
    state_preds=le.inverse_transform(np.argmax(ens,axis=1))
    print("\n[Training] Intensity Regressor...")
    reg=XGBRegressor(n_estimators=400,max_depth=4,learning_rate=0.07,subsample=0.8,random_state=42,n_jobs=-1)
    reg.fit(X_tr,yi,sample_weight=sw)
    reg_cv=cross_val_score(reg,X_tr,yi,cv=5,scoring='neg_mean_absolute_error')
    int_preds=np.clip(np.round(reg.predict(X_te)).astype(int),1,5)
    print(f"  Intensity CV MAE: {-reg_cv.mean():.3f} ± {reg_cv.std():.3f}")
    print("  Approach: Regression (XGBoost) — ordinal ordering preserved.")
    print("\n[Decision + Uncertainty]...")
    results=[]
    for i,(_,row) in enumerate(df_te.iterrows()):
        state=state_preds[i]; what,when,msg=decide(state,row)
        conf,flag=compute_unc(row['journal_text'],row.get('reflection_quality','vague'),ens[i])
        results.append({'id':row['id'],'predicted_state':state,
            'predicted_intensity':int(int_preds[i]),'confidence':conf,
            'uncertain_flag':flag,'what_to_do':what,'when_to_do':when,'support_message':msg})
    df_out=pd.DataFrame(results); df_out.to_csv(out_path,index=False)
    print(f"\n[Output] → {out_path}")
    print(pd.Series(state_preds).value_counts().to_string())
    print(f"Uncertain: {df_out['uncertain_flag'].sum()}/{len(df_out)} | Avg conf: {df_out['confidence'].mean():.3f}")
    print("\n[Feature Importance] Top 20 (XGBoost):")
    fi=pd.Series(xgb.feature_importances_,index=X_tr.columns).nlargest(20)
    vocab={f'tfidf_{i}':w for i,w in enumerate(tfidf.get_feature_names_out())}
    for f,v in fi.items(): print(f"  {vocab.get(f,f):<35} {v:.4f}")
    print("\n[Ablation Study]")
    tc=[c for c in X_tr.columns if c.startswith('tfidf_') or c in
        ['pos_score','neg_score','mixed_score','sentiment','word_count','is_short','has_contrast','uncertainty_score','is_vague_phrase']]
    mc=[c for c in X_tr.columns if c in
        ['sleep_hours','energy_level','stress_level','duration_min','time_of_day_enc',
         'ambience_enc','prev_mood_enc','face_emotion_enc','reflection_quality_enc','energy_stress_ratio','low_sleep','high_stress']]
    for lbl,cols in [('Text-Only',tc),('Meta-Only',mc),('Text+Meta',list(X_tr.columns))]:
        use=[c for c in cols if c in X_tr.columns]
        if not use: continue
        tmp=XGBClassifier(n_estimators=200,max_depth=4,use_label_encoder=False,eval_metric='mlogloss',random_state=42,n_jobs=-1)
        sc=cross_val_score(tmp,X_tr[use],y,cv=cv,scoring='accuracy')
        print(f"  {lbl:<20} Acc:{sc.mean():.3f}±{sc.std():.3f} ({len(use)} feats)")
    pickle.dump({'xgb':xgb,'rf':rf,'lr':lr,'reg':reg,'tfidf':tfidf,'imputer':imp,'le':le},
        open(os.path.join(BASE,'models/models.pkl'),'wb'))
    print("\n[Models] Saved → models/models.pkl\n✅ Pipeline complete!")
    return df_out

if __name__=='__main__':
    run_pipeline()
