CUDA_VISIBLE_DEVICES=3 /common/users/wq50/envs/pace/bin/python \
/common/users/wq50/SAE_path/concept_steer/area_counterfactual_spatial_anchor_sweep.py \
  --slide-selection borderline \
  --n-slides-total 10 \
  --area-side 7 \
  --high-attn-fraction 0.30 \
  --strengths 0.00,0.35,0.70,1.00 \
  --directions to_hpv_pos,to_hpv_neg \
  --identity-at-zero \
  --norm-match-to-orig
