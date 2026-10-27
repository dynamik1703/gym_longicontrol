import numpy as np
import pytest
import torch

from benchmarks.bankers_masked_rl_pilot_v3.masked_distribution import distribution
from benchmarks.bankers_masked_rl_pilot_v3.sac_agent import DiscreteSAC, targets
from benchmarks.bankers_masked_rl_pilot_v3.shield_treatments import METHODS
from gym_longicontrol.domain.state import VehicleState
from test_verifier import context


def test_post_keeps_unsafe_proposal_in_replay():
    r,audit=context(METHODS[1],[0],[20],VehicleState(velocity_m_s=5.5))
    prepared=r.prepare()
    prepared['proposal']=20
    assert not r.physical_mask[20]
    row=r.advance(prepared=prepared,full_evidence=True,learn=False)
    audit.verify(row,r.environment.unwrapped.track,METHODS[1],full=True)
    assert row['proposal']==20 and row['executed']!=20
    assert r.replay.arrays['action'][0,0]==20
    assert r.replay.arrays['executed'][0,0]==row['executed']
    assert row['replacement_reason']=='PROPOSAL_NOT_CERTIFIED'


def test_successor_support_and_invalid_q_expectations():
    current=torch.ones((1,21),dtype=torch.bool)
    successor=torch.zeros((1,21),dtype=torch.bool)
    successor[0,0]=True
    logits=torch.zeros((1,21))
    p,lp,entropy,target=distribution(logits,successor)
    q=torch.full((1,21),1e20)
    q[0,0]=2
    costs=torch.full((1,3,21),1e20)
    costs[:,:,0]=3
    yr,yc=targets(torch.tensor([[1.0]]),torch.zeros((1,3)),torch.tensor([[0.5]]),
                  p,lp,q,costs,1.0)
    assert yr.item()==2 and torch.equal(yc,torch.full((1,3),1.5))
    assert entropy.item()==0 and target.item()==0
    assert distribution(logits,current)[2].item()>0


def test_singleton_optimizer_finite():
    agent=DiscreteSAC(11)
    mask=np.zeros((256,21),dtype=np.bool_)
    mask[:,0]=True
    batch=dict(observation=np.zeros((256,10),dtype=np.float32),
        next_observation=np.zeros((256,10),dtype=np.float32),action=np.zeros((256,1),dtype=np.int64),
        reward=np.zeros((256,1),dtype=np.float32),costs=np.zeros((256,3),dtype=np.float32),
        discount=np.full((256,1),0.99**0.1,dtype=np.float32),mask=mask,next_mask=mask)
    result=agent.update(batch)
    assert all(np.isfinite(x) for x in result.values())
    assert result['entropy']==0 and result['target_entropy']==0


@pytest.mark.parametrize('proposal',range(1,20))
def test_all_native_grid_lower_ties(proposal):
    # Restrict admission to symmetric neighbors to isolate the selector;
    # retain genuine candidate and selected production certificates.
    r,_=context(METHODS[1],[0],[100])
    r.reset()
    mask=[False]*21
    mask[proposal-1]=mask[proposal+1]=True
    t=r.treatment
    # Reuse genuine certs for the two admitted actions; the selector's final
    # production check still runs normally and must agree with its chosen cert.
    certificates=r.certificates
    selected,certificate,reason=t.execute(proposal,r.ordinary,r.memory,mask,certificates,r.binding)
    assert selected==proposal-1 and certificate['accepted']
