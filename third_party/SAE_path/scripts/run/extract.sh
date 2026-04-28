for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i /common/users/wq50/envs/pace/bin/python /common/users/wq50/SAE_path/scripts/extract_filtered_uni_from_wsi.py \
    --wsi_dir /common/users/wq50/SAE_path/wsi/hnsc_hpv \
    --out_dir /common/users/wq50/SAE_path/extracted_features/hnsc_hpv_filtered_uni2h \
    --summary_dir /common/users/wq50/SAE_path/extracted_features/hnsc_hpv_filtered_uni2h_summary \
    --coords_dir /common/users/wq50/SAE_path/extracted_features/hnsc_hpv_filtered_uni2h_coords \
    --viz_dir /common/users/wq50/SAE_path/extracted_features/hnsc_hpv_filtered_uni2h_viz \
    --reader auto \
    --device cuda:0 \
    --batch_size 1024 \
    --filter_workers 4 \
    --num_shards 4 \
    --shard_index $i \
    > /common/users/wq50/SAE_path/extracted_features/hnsc_hpv_filtered_uni2h_summary/shard_${i}.log 2>&1 &
done
wait
