python ~/code/pointgoalnav_unsup_rgbd/vo_training.py \
--device cuda:2 \
--run_type train_no_noise \
--train_epoch 20 \
--lr 1.e-4 \
--dof 3 \
--train_scale 1 \
--rec_loss_coef 2.0 \
--num_frame 3 \
--pose_inputs rgbd \
--rec_target texture \
--classification_model \
--log_name vo_irgbd_ttexture_3frame_cmodel_rmsprop_20epoch \
# --reuse \
# --reuse_name vo_irgbd_trgbd_3frame_cmodel \
# --act_spec \
# --reuse
# --loop_loss \
# --classification_model \