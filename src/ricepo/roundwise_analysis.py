"""Matched-query descriptive retrieval analysis; no policy selection or generation."""
from statistics import mean


def _index(rows):
    indexed={r['query_id']:r for r in rows}
    if not rows or len(indexed)!=len(rows):
        raise ValueError('empty or duplicate query identity')
    return indexed


def _mean(values):
    values=list(values)
    return mean(values) if values else None


def summarize(rows):
    _index(rows)
    complete=[r for r in rows if not r['failed']]
    depths=[]
    for d in range(1,1+max((len(r['rounds']) for r in rows),default=0)):
        reached=[r for r in rows if len(r['rounds'])>=d]
        depths.append(dict(depth=d,count=len(reached),matched_d0=_mean(r['d0'] for r in reached),
                           ndcg=_mean(r['rounds'][d-1]['ndcg'] for r in reached)))
    declines={}
    for r in rows:
        previous=r['d0']
        for turn in r['rounds']:
            if turn['ndcg']<previous:
                key=str(turn['depth']);declines[key]=declines.get(key,0)+1
                break
            previous=turn['ndcg']
    return dict(query_count=len(rows),complete_count=len(complete),failed_count=len(rows)-len(complete),
        mean_d0=_mean(r['d0'] for r in rows),mean_task_score=_mean(r['task_score'] for r in rows),
        mean_last_valid_ndcg=_mean(r['last_valid_ndcg'] for r in rows),
        d0_minus_task=_mean(r['d0']-r['task_score'] for r in rows),
        d0_minus_last_valid=_mean(r['d0']-r['last_valid_ndcg'] for r in rows),
        failure_utility_gap=_mean(r['last_valid_ndcg']-r['task_score'] for r in rows),
        complete_only=dict(count=len(complete),matched_d0=_mean(r['d0'] for r in complete),
                           final_ndcg=_mean(r['rounds'][-1]['ndcg'] for r in complete)),
        first_decline_depth_counts=declines,depths=depths,
        intermediate_better_than_final_count=sum(bool(r['rounds'][:-1]) and
            max(t['ndcg'] for t in r['rounds'][:-1])>r['rounds'][-1]['ndcg'] for r in complete),
        repeated_summary_transitions=sum(a['summary_sha256']==b['summary_sha256']
            for r in rows for a,b in zip(r['rounds'],r['rounds'][1:])),
        repeated_ranking_transitions=sum(a['ranking_sha256']==b['ranking_sha256']
            for r in rows for a,b in zip(r['rounds'],r['rounds'][1:])))


def paired_divergence(left,right):
    a,b=_index(left),_index(right)
    if a.keys()!=b.keys():raise ValueError('paired query membership differs')
    result=[]
    for q in sorted(a):
        x,y=a[q],b[q]
        if x['d0']!=y['d0']:raise ValueError('paired D0 differs')
        pairs=list(zip(x['rounds'],y['rounds']))
        def first(key):
            return next((u['depth'] for u,v in pairs if u[key]!=v[key]),None)
        result.append(dict(query_id=q,d0=x['d0'],left_task=x['task_score'],right_task=y['task_score'],
            first_summary_difference_depth=first('summary_sha256'),
            first_ranking_difference_depth=first('ranking_sha256'),first_ndcg_difference_depth=first('ndcg'),
            left_valid_rounds=len(x['rounds']),right_valid_rounds=len(y['rounds']),
            left_failed=x['failed'],right_failed=y['failed']))
    return result
