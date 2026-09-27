"""Causal formal scheduling: repeat gain, marginal coverage, and tag confirmation.

Current weather is held across exposure segments; hidden efficiency/tag effects
are estimated from past realized feedback. No scenario/evaluator access.
"""
from collections import Counter,defaultdict,deque
from datetime import datetime,timezone
import math
import os
import statistics
from public_geometry import Tile,geometry_sample

# Selected on the complete development pair, frozen before new holdouts.
DEFAULTS={'ENABLE':1,'CALIBRATE':0,'COVERAGE':0,'INFORMATION':1}

def stamp(value): return datetime.fromisoformat(str(value).replace('Z','+00:00')).timestamp()

def evenness(counts):
    squares=sum(x*x for x in counts)
    return sum(counts)**2/(len(counts)*squares) if squares else 0.0

def coverage_delta(weight,total_base,base_delta,counts,region_index,first):
    new=list(counts)
    if first: new[region_index]+=1
    return weight*((total_base+base_delta)*evenness(new)-total_base*evenness(counts))

class FormalOptimizer:
    def __init__(self,initial):
        self.enabled=bool(int(os.getenv('SAC_FORMAL_ENABLE',str(DEFAULTS['ENABLE']))))
        self.calibrate=bool(int(os.getenv('SAC_FORMAL_CALIBRATE',str(DEFAULTS['CALIBRATE']))))
        self.coverage=bool(int(os.getenv('SAC_FORMAL_COVERAGE',str(DEFAULTS['COVERAGE']))))
        self.information=bool(int(os.getenv('SAC_FORMAL_INFORMATION',str(DEFAULTS['INFORMATION']))))
        self.initial=initial
        self.catalog={r['tile_id']:r for r in initial.get('tile_catalog',{}).get('tiles',[])}
        self.regions=sorted({r.get('region_id','') for r in self.catalog.values()})
        self.config=initial['scoring_contract']['score_config']
        self.samples=defaultdict(lambda:deque(maxlen=9))
        self.normal=deque(maxlen=64)
        self.bests={}; self.base_bests={}; self.pending=None
        self.estimates={}; self.last_diagnostics={}
        self.search=None
        if os.getenv('SAC_FORMAL_SEARCH','off')=='beam':
            from formal_search import FormalSearch
            self.search=FormalSearch(self)

    def update(self,snapshot):
        pending,self.pending=self.pending,None
        feedback=snapshot.get('tile_last_finished')
        if not pending or not feedback or feedback.get('tile_id')!=pending['tile_id']: return
        score=float(feedback.get('score',0)); tile=pending['tile_id']
        if score>self.bests.get(tile,0):
            self.bests[tile]=score; self.base_bests[tile]=score*pending['base_share']
        if score<=0 or pending['expected']<=0 or pending['confounded']: return
        ratio=max(.05,min(2.5,score/pending['expected']))
        self.samples[tile].append(ratio)
        if .65<=ratio<=1.15: self.normal.append(ratio)

    def factor(self,tile):
        if not self.calibrate: return 1.0
        common=statistics.median(self.normal) if self.normal else 1.0
        samples=self.samples[tile]
        if not samples: return common
        return (2*common+len(samples)*statistics.median(samples))/(2+len(samples))

    def exposure(self,raw,program,snapshot):
        now=stamp(snapshot['cursor']['timestamp_utc']); origin=now-int(snapshot['cursor']['slot_offset_seconds'])
        duration=int(raw['nominal_exptime_seconds']); end=now+duration
        if now<stamp(raw['window_start_utc']) or end>stamp(raw['window_end_utc']): return None
        w=raw['effective_weather']
        if not w['is_observable']: return None
        tile=self.catalog[raw['tile_id']]
        coords=Tile(float(tile['ra_deg']),float(tile['dec_deg']))
        contract=self.initial['scoring_contract']; interface=contract['weather_score_interface']
        multiplier=float(w['transparency'])*float(w['sky_quality'])/float(w['seeing_arcsec'])
        base=bonus=0.0; cursor=now
        while cursor<end:
            stop=min(end,origin+(math.floor((cursor-origin)/900)+1)*900)
            geom=geometry_sample(coords,datetime.fromtimestamp((cursor+stop)/2,timezone.utc),{'lunar_model':contract['lunar_model']},{'site':self.initial['site']})
            start_geom=geometry_sample(coords,datetime.fromtimestamp(cursor,timezone.utc),{'lunar_model':contract['lunar_model']},{'site':self.initial['site']})
            if min(geom['altitude_deg'],start_geom['altitude_deg'])<30: return None
            q=min(multiplier/geom['airmass']**float(interface['airmass_exponent']),float(interface['maximum_weather_quality']))*geom['lunar_quality_factor']
            threshold=self.config['quality_thresholds']
            band='DARK' if q>=threshold['dark'] else 'BRIGHT' if q>=threshold['bright'] else 'BACKUP'
            part=float(raw['tile_science_value'])*q*(stop-cursor)/duration
            base+=part; bonus+=part*self.config['program_bonus'][program] if band==program else 0
            cursor=stop
        return base,bonus

    def information_value(self,row,snapshot,detector):
        if not self.information: return 0.0
        reads=detector._tag_reads.get(row.tile_id,[])
        if not reads or reads[-1][0] is None: return 0.0
        tag=reads[-1][0]
        if (row.tile_id,tag) in detector._reported_tags: return 0.0
        hits=[night for band,night in reads if band==tag]
        if len(hits)/len(reads)<.6: return 0.0
        if len(reads)>=detector.tag_min_reads and snapshot['cursor']['night_id'] in hits: return 0.0
        remaining=max(1,detector.tag_min_reads-len(reads))
        # A bounded heuristic information budget, not a calibrated probability.
        reward=float(self.config.get('reporting',{}).get('reward_correct',100))
        return min(20.0,.3*reward/remaining)

    def choose(self,snapshot,previews,default,detector):
        if not self.enabled: return default
        raws={r['tile_id']:r for r in snapshot['candidate_tiles']}
        completed=set(snapshot.get('progress',{}).get('completed_tile_ids',[]))
        counts=Counter(self.catalog[t]['region_id'] for t in completed if t in self.catalog)
        vector=[counts[r] for r in self.regions]
        total_base=sum(self.base_bests.values())
        requests={r['request_id']:r for r in snapshot.get('active_requests',[])}
        now=stamp(snapshot['cursor']['timestamp_utc'])
        self.estimates={}; ranked=[]
        for row in previews:
            raw=raws[row.tile_id]
            request=requests.get(row.request_id)
            if row.request_id and (request is None or now<stamp(request['available_from_utc']) or now+row.nominal_exptime_seconds>stamp(request['deadline_utc'])): continue
            prediction=self.exposure(raw,row.program,snapshot)
            if prediction is None: continue
            base,bonus=prediction; public_total=base+bonus
            self.estimates[(row.tile_id,row.program,row.request_id)]=prediction
            factor=self.factor(row.tile_id); expected=public_total*factor
            old=self.bests.get(row.tile_id,detector.bests.get(row.tile_id,0))
            science=max(0,expected-old)
            base_gain=base*factor-self.base_bests.get(row.tile_id,0) if expected>old else 0
            coverage=coverage_delta(float(self.config.get('coverage_bonus_weight',0)),total_base,base_gain,vector,self.regions.index(row.region_id),row.tile_id not in completed) if self.coverage and self.regions else 0
            # Request reward remains the inherited per-remaining-tile estimate;
            # distribute over remaining visits rather than awarding each full credit.
            request_credit=row.request_policy_value
            if request:
                requirement=next((v for v in request.get('tile_requirements',[]) if v['tile_id']==row.tile_id),{})
                request_credit/=max(1,int(requirement.get('remaining_visits',1)))
            info=self.information_value(row,snapshot,detector)
            utility=science+row.terminal_penalty_avoidance+request_credit+coverage+info
            ranked.append((utility/row.nominal_exptime_seconds,science/row.nominal_exptime_seconds,row,{'science':science,'coverage':coverage,'information':info,'factor':factor}))
        if not ranked: return {'action':'wait','tile_id':'','program':'','request_id':'','reason':'formal optimizer has no legal complete exposure','decision_source':'formal_optimizer'}
        _,_,row,details=max(ranked,key=lambda x:(x[0],x[1]))
        self.last_diagnostics=details
        decision={'action':'observe','tile_id':row.tile_id,'program':row.program,'request_id':row.request_id,'reason':f'formal gain calibration={int(self.calibrate)} coverage={int(self.coverage)} information={int(self.information)}','decision_source':'formal_optimizer'}
        return self.search.choose(snapshot,ranked,decision,detector) if self.search else decision

    def note(self,decision,snapshot,detector):
        key=tuple(decision.get(k,'') for k in ['tile_id','program','request_id'])
        prediction=self.estimates.get(key)
        if decision['action']!='observe' or prediction is None:
            self.pending=None; return None
        base,bonus=prediction; expected=base+bonus
        raw=next(r for r in snapshot['candidate_tiles'] if r['tile_id']==key[0])
        confounded=detector.under_cold_wave(snapshot) or detector._in_fault_scope(raw)
        self.pending={'tile_id':key[0],'expected':expected,'base_share':base/expected if expected else 0,'confounded':confounded}
        return expected
