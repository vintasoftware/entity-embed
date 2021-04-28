.. _cli:

======================
Command Line Interface
======================

Commands
--------

After installing entity-embed with pip, the commands ``entity_embed_train`` and ``entity_embed_predict`` are available.

Train Options
~~~~~~~~~~~~~

To check all the CLI options for train, run:

.. code-block:: bash

    $ entity_embed_train --help

The **mandatory** ones are:

.. csv-table::
   :file: cli_train_options.csv
   :widths: 30, 70
   :header-rows: 1

If you're doing Record Linkage, there are other **mandatory** options:

.. csv-table::
   :file: cli_rl_options.csv
   :widths: 30, 70
   :header-rows: 1

Predict Options
~~~~~~~~~~~~~~~

To check all the CLI options for predict, run:

.. code-block:: bash

    $ entity_embed_predict --help

The **mandatory** ones are:

.. csv-table::
   :file: cli_predict_options.csv
   :widths: 30, 70
   :header-rows: 1

If you're doing Record Linkage, there are other **mandatory** options:

.. csv-table::
   :file: cli_rl_options.csv
   :widths: 30, 70
   :header-rows: 1

Examples
--------

#. Clone entity-embed GitHub repo: ``git clone https://github.com/vintasoftware/entity-embed.git``
#. ``cd`` into it
#. Check example data in ``example-data/``
#. Back to entity-embed root dir, run one of the following:

Deduplication Train
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    $ entity_embed_train \
        --field_config_json example-data/er-field-config.json \
        --train_csv example-data/er-train.csv \
        --valid_csv example-data/er-valid.csv \
        --test_csv example-data/er-test.csv \
        --unlabeled_csv example-data/er-unlabeled.csv \
        --csv_encoding utf-8 \
        --cluster_field cluster \
        --embedding_size 300 \
        --lr 0.001 \
        --min_epochs 5 \
        --max_epochs 100 \
        --early_stop_monitor valid_recall_at_0.3 \
        --early_stop_min_delta 0 \
        --early_stop_patience 20 \
        --early_stop_mode max \
        --tb_save_dir tb_logs \
        --tb_name er-example \
        --check_val_every_n_epoch 1 \
        --batch_size 32 \
        --eval_batch_size 64 \
        --num_workers -1 \
        --multiprocessing_context fork \
        --sim_threshold 0.3 \
        --sim_threshold 0.5 \
        --sim_threshold 0.7 \
        --ann_k 100 \
        --m 64 \
        --max_m0 64 \
        --ef_construction 150 \
        --ef_search -1 \
        --random_seed 42 \
        --model_save_dir trained-models/er/ \
        --use_gpu 1

Deduplication Predict
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    $ entity_embed_predict \
        --model_save_filepath "trained-models/er/...fill-here..." \
        --unlabeled_csv example-data/er-unlabeled.csv \
        --csv_encoding utf-8 \
        --eval_batch_size 50 \
        --num_workers -1 \
        --multiprocessing_context fork \
        --sim_threshold 0.3 \
        --ann_k 100 \
        --m 64 \
        --max_m0 64 \
        --ef_construction 150 \
        --ef_search -1 \
        --random_seed 42 \
        --output_json example-data/er-prediction.json \
        --use_gpu 1

Record Linkage Train
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    $ entity_embed_train \
        --field_config_json example-data/rl-field-config.json \
        --train_csv example-data/rl-train.csv \
        --valid_csv example-data/rl-valid.csv \
        --test_csv example-data/rl-test.csv \
        --unlabeled_csv example-data/rl-unlabeled.csv \
        --csv_encoding utf-8 \
        --cluster_field cluster \
        --source_field __source \
        --left_source amazon \
        --embedding_size 300 \
        --lr 0.001 \
        --min_epochs 5 \
        --max_epochs 100 \
        --early_stop_monitor valid_recall_at_0.3 \
        --early_stop_min_delta 0 \
        --early_stop_patience 20 \
        --early_stop_mode max \
        --tb_save_dir tb_logs \
        --tb_name rl-example \
        --check_val_every_n_epoch 1 \
        --batch_size 32 \
        --eval_batch_size 64 \
        --num_workers -1 \
        --multiprocessing_context fork \
        --sim_threshold 0.3 \
        --sim_threshold 0.5 \
        --sim_threshold 0.7 \
        --ann_k 100 \
        --m 64 \
        --max_m0 64 \
        --ef_construction 150 \
        --ef_search -1 \
        --random_seed 42 \
        --model_save_dir trained-models/rl/ \
        --use_gpu 1

Record Linkage Predict
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    $ entity_embed_predict \
        --model_save_filepath "trained-models/rl/...fill-here..." \
        --unlabeled_csv example-data/rl-unlabeled.csv \
        --csv_encoding utf-8 \
        --source_field __source \
        --left_source amazon \
        --eval_batch_size 50 \
        --num_workers -1 \
        --multiprocessing_context fork \
        --sim_threshold 0.3 \
        --ann_k 100 \
        --m 64 \
        --max_m0 64 \
        --ef_construction 150 \
        --ef_search -1 \
        --random_seed 42 \
        --output_json example-data/rl-prediction.json \
        --use_gpu 1
