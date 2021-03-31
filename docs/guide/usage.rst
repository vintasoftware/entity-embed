=====
Usage
=====

This guide will teach you how to train Entity Embed's deep neural network, use it to embed entities, index those embedding vectors into an Approximate Nearest Neighbors index, and finally search for candidate duplicate pairs on that index. All using the Python API.

It's also possible to run Entity Embed from the command line, check the :ref:`Command Line Interface <cli>` guide.

Deduplication
-------------

Let's learn first how to perform Deduplication on Entity Embed. After that, we can learn the specifics on how to perform :ref:`Record Linkage`. Deduplication means finding duplicates within a single dataset. Record Linkage means finding matching rows across two datasets.

Preparing the data
~~~~~~~~~~~~~~~~~~

Your data needs to represent a list of entities as list of ``dict`` s. Those ``dict`` s must contain:

* an ``id`` to uniquely identify each entity
* a ``cluster`` key that indicates the true matching entities
* the fields you want to use for embedding

For example::

    [{'id': 0,
      'cluster': 0,
      'title': '001-Berimbou',
      'artist': 'Astrud Gilberto',
      'album': 'Look to the Rainbow (2008)'},
     {'id': 1,
      'cluster': 0,
      'title': 'Berimbau',
      'artist': 'Astrud Gilberto',
      'album': 'Look to the Rainbow (1966)'},
     {'id': 2,
      'cluster': 1,
      'title': '4 - Marooned - Pink Floyd',
      'artist': '',
      'album': 'The Division Bell'}]

That's called a ``row_dict`` across Entity Embed's API. Once you have your ``row_dict``, you must split it into train, valid, and test data::

    from entity_embed.data_utils import utils

    train_row_dict, valid_row_dict, test_row_dict = utils.split_row_dict_on_clusters(
        row_dict=row_dict,
        cluster_attr="cluster",
        train_proportion=0.6,
        valid_proportion=0.2,
        random_seed=42,
    )

Note we're splitting the data on **clusters**, not rows, so the row counts vary across the returned ``row_dict`` s.

Defining the fields
~~~~~~~~~~~~~~~~~~~

We need to define how entity fields will be numericalized and encoded by Entity Embed's deep neural network. First, we need an ``alphabet`` . The default alphabet has the ASCII numbers, letters, symbols and space. You can use any other alphabet if you need::

    from entity_embed.data_utils.attr_config_parser import DEFAULT_ALPHABET

Then we define an ``attr_config_dict`` . It defines :ref:`Field Types <field_types>` that determine how fields are processed in the neural network::

    attr_config_dict = {
        'title': {
            'field_type': "MULTITOKEN",
            'tokenizer': "entity_embed.default_tokenizer",
            'alphabet': DEFAULT_ALPHABET,
            'max_str_len': None,  # compute
        },
        'title_semantic': {
            'source_attr': 'title',
            'field_type': "SEMANTIC_MULTITOKEN",
            'tokenizer': "entity_embed.default_tokenizer",
            'vocab': "fasttext.en.300d",
        },
        'artist': {
            'field_type': "MULTITOKEN",
            'tokenizer': "entity_embed.default_tokenizer",
            'alphabet': DEFAULT_ALPHABET,
            'max_str_len': None,  # compute
        },
        'album': {
            'field_type': "MULTITOKEN",
            'tokenizer': "entity_embed.default_tokenizer",
            'alphabet': DEFAULT_ALPHABET,
            'max_str_len': None,  # compute
        },
        'album_semantic': {
            'source_attr': 'album',
            'field_type': "SEMANTIC_MULTITOKEN",
            'tokenizer': "entity_embed.default_tokenizer",
            'vocab': "fasttext.en.300d",
        }
    }

.. note::
    Check the available :ref:`Field Types <field_types>` and use the ones that make most sense for your data.

With the ``attr_config_dict``, we can get a ``row_numericalizer`` . This object will convert the strings from our entities into tensors for the neural network::


    from entity_embed import AttrConfigDictParser

    row_numericalizer = AttrConfigDictParser.from_dict(attr_config_dict, row_list=row_dict.values())

.. warning::
    Note the ``attr_config_dict`` receives a ``row_list`` . Here we're passing ``row_list=row_dict.values()``, meaning we're passing all train, valid, and test data. **If you have unlabeled data, you should include it too in** ``row_list`` . It's important to build the ``row_numericalizer`` with ALL available data, labeled or not. This ensures numericalization will know the true ``max_str_len`` of the fields of your data, and the true vocabulary of tokens to generalize well.

Building the model
~~~~~~~~~~~~~~~~~~

Under the hood, Entity Embed uses `pytorch-lightning <https://pytorch-lightning.readthedocs.io/en/latest/>`_, so we need to create a datamodule object::

    from entity_embed import DeduplicationDataModule

    datamodule = DeduplicationDataModule(
        train_row_dict=train_row_dict,
        valid_row_dict=valid_row_dict,
        test_row_dict=test_row_dict,
        cluster_attr="cluster",
        row_numericalizer=row_numericalizer,
        batch_size=32,
        eval_batch_size=64,
        random_seed=42,
    )

Training the model
~~~~~~~~~~~~~~~~~~

Now the training process!

We must choose the K of the Approximate Nearest Neighbors, i.e., the top K neighbors our model will use to find duplicates in the embedding space. Below we're using the ``row_numericalizer`` and ``ann_k`` to initializing the ``EntityEmbed`` model object::

    from entity_embed import EntityEmbed

    model = EntityEmbed(
        row_numericalizer,
        ann_k=100,
    )

To train, Entity Embed uses `pytorch-lightning Trainer <https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html>`_ on it's ``EntityEmbed.fit`` method.

Since Entity Embed is focused in recall, we'll use ``valid_recall_at_0.3`` for early stopping. But we'll set ``min_epochs = 5`` to avoid a very low precision.

``0.3`` here is the threshold for cosine similarity of embedding vectors, so possible values are between -1 and 1. We're using a validation metric, and the training process will run validation on every epoch end due to ``check_val_every_n_epoch=1`` .

We also set ``tb_name`` and ``tb_save_dir`` to use Tensorboard. Run ``tensorboard --logdir notebooks/tb_logs`` to check the train and valid metrics during and after training::

    trainer = model.fit(
        datamodule,
        min_epochs=5,
        max_epochs=100,
        check_val_every_n_epoch=1,
        early_stop_monitor="valid_recall_at_0.3",
        tb_save_dir='tb_logs',
        tb_name='music',
    )

``EntityEmbed.fit`` keeps only the weights of the best validation model. With them, we can check the best performance on validation set::

    model.validate(datamodule)

And we can check which fields are most important for the final embedding::

    model.get_signature_weights()

Again with the best validation model, we can check the performance on the test set::

    model.test(datamodule)

Indexing embeddings / Production run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When running in production, you only have access to the trained ``model`` object and the production ``row_dict`` (without the true clusters filled, of course). You can get the embedding vectors of a production ``row_dict`` using the ``predict`` method::

    vector_dict = model.predict(
        row_dict=production_row_dict,
        batch_size=64
    )

The ``vector_dict`` maps ``id`` s to numpy arrays. We can build an `ANNEntityIndex`, insert all embeddings from `vector_dict` on it, and build it::

    from entity_embed import ANNEntityIndex

    ann_index = ANNEntityIndex(embedding_size=model.embedding_size)
    ann_index.insert_vector_dict(vector_dict)
    ann_index.build()

With the index built, we can now search on it and find the candidate duplicate pairs::

    found_pair_set = ann_index.search_pairs(
        k=100,
        sim_threshold=0.3,
    )

.. note::
    Even though we used the same ``k`` and one of the ``sim_threshold`` s from the model training, you're free to use any value you want here.

``found_pair_set`` is a set of tuple ``id`` pairs with the smaller ``id`` always on the first position of the tuple.

You must filter the ``found_pair_set`` to find the best matching pairs. One option is to use pairwise classifiers like the ones from `Python Record Linkage Toolkit <https://recordlinkage.readthedocs.io/en/latest/index.html>`_ .

Record Linkage
--------------

The steps to perform Record Linkage are similar to the ones for :ref:`Deduplication`, but you must provide additional parameters and use different classes. Below we highlight only the differences:

Preparing the data
~~~~~~~~~~~~~~~~~~

On your data for Record Linkage, you must include a field on each entity to inform what is its source dataset. For example::


    [{'id': 0,
      'cluster': 0,
      '__source': "left",
      'title': '001-Berimbou',
      'artist': 'Astrud Gilberto',
      'album': 'Look to the Rainbow (2008)'},
     {'id': 1,
      'cluster': 0,
      '__source': "right",
      'title': 'Berimbau',
      'artist': 'Astrud Gilberto',
      'album': 'Look to the Rainbow (1966)'},
     {'id': 2,
      'cluster': 1,
      '__source': "left",
      'title': '4 - Marooned - Pink Floyd',
      'artist': '',
      'album': 'The Division Bell'}]

.. warning::
    Currently Entity Embed only supports Record Linkage of two datasets at one time. On the example above, we have only two sources: ``"left"`` and ``"right"`` .

Building the model
~~~~~~~~~~~~~~~~~~

Use the ``LinkageDataModule`` class to initialize the ``datamodule`` . Note there are two additional parameters here: ``source_attr`` and ``left_source``::

    from entity_embed import LinkageDataModule

    datamodule = LinkageDataModule(
        train_row_dict=train_row_dict,
        valid_row_dict=valid_row_dict,
        test_row_dict=test_row_dict,
        source_attr="__source",
        left_source="left",
        cluster_attr="cluster",
        row_numericalizer=row_numericalizer,
        batch_size=32,
        eval_batch_size=64,
        random_seed=42,
    )

Training the model
~~~~~~~~~~~~~~~~~~

Use the ``LinkageEmbed`` class to initialize the model object. Again, there are two additional parameters here: ``source_attr`` and ``left_source``::

    from entity_embed import LinkageEmbed

    model = LinkageEmbed(
        row_numericalizer,
        ann_k=100,
        source_attr="__source",
        left_source="left",
    )

Indexing embeddings / Production run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When calling ``predict``, you will now get two ``vector_dict`` s, one for each source dataset::

    test_left_vector_dict, test_right_vector_dict = model.predict(
        row_dict=test_row_dict,
        batch_size=eval_batch_size
    )

Now init an `ANNLinkageIndex`, insert all embeddings from both `test_left_vector_dict` and `test_right_vector_dict` on it, and build it::

    from entity_embed import ANNLinkageIndex

    ann_index = ANNLinkageIndex(embedding_size=model.embedding_size)
    ann_index.insert_vector_dict(
        left_vector_dict=test_left_vector_dict,
        right_vector_dict=test_right_vector_dict,
    )
    ann_index.build()

With the index built, we can now search on it and find the candidate duplicate pairs::

    found_pair_set = ann_index.search_pairs(
        k=ann_k,
        sim_threshold=0.3,
        left_vector_dict=test_left_vector_dict,
        right_vector_dict=test_right_vector_dict,
        left_source=left_source,
    )

Here, ``found_pair_set`` is again a set of tuple ``id`` pairs, but there's a catch: the first position of the tuple will always have the left dataset ``id`` s, while the second position will have the right dataset ``id`` s.

Examples
--------

Check these Jupyter Notebooks for step-by-step examples:

- Deduplication, when you have a single dirty dataset with duplicates: `notebooks/Deduplication-Example.ipynb <https://github.com/vintasoftware/entity-embed/blob/main/notebooks/Deduplication-Example.ipynb>`_
- Record Linkage, when you have multiple clean datasets you need to link: `notebooks/Record-Linkage-Example.ipynb <https://github.com/vintasoftware/entity-embed/blob/main/notebooks/Record-Linkage-Example.ipynb>`_
