python -m real_robot.train \
  agents=ddim_agent \
  agents/model=ddim/ddim_dec_transformer \
  agents/obs_encoders=resnet \
  agents.if_film_condition=False \
  task_name=pick_up_the_orange_and_place_it_in_the_basket \
  wandb.mode=offline \
  scaler_type=standard \
  epoch=50 \
  train_batch_size=64 \
  val_batch_size=64 \
  num_workers=8