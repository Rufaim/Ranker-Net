import tensorflow as tf
from data_importer import DataImporter, DetectionsLabels
from rank_net_v2 import RankNetworkV2

from layers import NALU, GLU, Dense


INPUT_LEN = len(DataImporter.feature_columns)
NET_STRUCTURE = [NALU(28), Dense(20,tf.nn.relu)]
LEARNING_RATE = 1e-4
BATCH_SIZE = 1000
ITERATIONS = 100000

CHECKPOINT_DIR = "./checkpoints/"
CHECKPOINT_NAME = "model"
CHECKPOINT_ITER = 50



data_importer = DataImporter("../train_data/data.csv")
data_importer.load_data()

alphas = [data_importer.data[data_importer.data.object_class==DetectionsLabels.DRONE.value].shape[0] / data_importer.data.shape[0],
            data_importer.data[data_importer.data.object_class!=DetectionsLabels.DRONE.value].shape[0] / data_importer.data.shape[0]]
print("ALPHAS: ", alphas)
model = RankNetworkV2(INPUT_LEN,NET_STRUCTURE,LEARNING_RATE,alphas)

conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True
conf.gpu_options.per_process_gpu_memory_fraction = 0.25

with tf.Session(config=conf) as sess:
    model.initialize(sess)

    saver = tf.train.Saver(max_to_keep=3)
    checkpoint_name = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if checkpoint_name is not None:
        print("Restoring model from checkpoint...")
        saver.restore(sess, checkpoint_name)

    train_batch,test_batch = data_importer.random_batch_benerator(BATCH_SIZE,ITERATIONS)
    try:
        for X_1,X_2,y in train_batch:
            model.train_step(X_1,X_2)

            step = model.get_global_step()

            if step % CHECKPOINT_ITER == 0:
                X_1,X_2,y = next(test_batch)
                loss, accuracy, loss_median = model.get_metrics(X_1,X_2)
                saver.save(sess,CHECKPOINT_DIR + CHECKPOINT_NAME,global_step=step)

                print("Step: {:10} | LR: {:10} | Loss: {:15} | LossMedian: {:15} | Accuracy {:3}".format(
                                            step, model.get_learning_rate(), loss, loss_median,
                                            accuracy ))

    except KeyboardInterrupt:
            print("Training interrupted!")
    finally:
        saver.save(sess,CHECKPOINT_DIR + CHECKPOINT_NAME,global_step=model.get_global_step())
        model.dump_network_to_file("net.json")
