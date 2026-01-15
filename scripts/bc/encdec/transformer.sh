python run.py  --config-name=libero_config \
            --multirun agents=bc_agent \
            agent_name=bc_transformer \
            group=bc_encoder_decoder \
            agents/model=bc/bc_encdec_transformer \
            task_suite=libero_object \
            traj_per_task=10 \
            seed=0,1,2