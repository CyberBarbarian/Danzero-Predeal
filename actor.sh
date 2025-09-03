# rm -rf /aiarena/nas/guandan_douzero/data/*
# rm -rf /aiarena/nas/guandan_douzero/mempool.pkl
# python3 -u /aiarena/nas/guandan_douzero/mempool.py > /aiarena/nas/guandan_douzero/mempool.log &
python3 train.py --gpu_devices 0,1,2,3 --num_actor_devices 3 --num_actors 2 --training_device 3 --total_frames 200000