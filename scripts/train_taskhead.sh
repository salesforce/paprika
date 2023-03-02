CUDA_VISIBLE_DEVICES=0 python engines/main_train_taskhead.py --cfg configs/downstream/mlp_coin_sf.yml --checkpoint /export/share/hongluzhou/data/checkpoints/PAFoMo_howto100m_fullset.pth

CUDA_VISIBLE_DEVICES=0 python engines/main_train_taskhead.py --cfg configs/downstream/mlp_coin_sf.yml --checkpoint /export/share/hongluzhou/data/checkpoints/DS_howto100m_fullset.pth

