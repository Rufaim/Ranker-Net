import tensorflow as tf
from data_importer import DataImporter, DetectionsLabels
from rank_net import RankNetwork

from layers import NALU, GLU, Dense

import csv
import sys
import argparse

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nn1', '--number_neurons_1', default = 28)
    parser.add_argument('-nn2', '--number_neurons_2', default = 20)

    return parser

parser = createParser()
program_params = parser.parse_args(sys.argv[1:])

RANDOM_SEED = 42
RANDOM_SEED_DATA = 124
INPUT_LEN = len(DataImporter.feature_columns)
NET_STRUCTURE = [   Dense(int(program_params.number_neurons_1),initializer=tf.contrib.layers.xavier_initializer(seed=RANDOM_SEED)), 
                    Dense(int(program_params.number_neurons_2),tf.nn.relu,initializer = tf.contrib.layers.xavier_initializer(seed=RANDOM_SEED))  ]
#NALU(int(program_params.number_neurons_1),initializer=tf.contrib.layers.xavier_initializer(seed=RANDOM_SEED)),

LEARNING_RATE = 1e-4
BATCH_SIZE = 1000
ITERATIONS = 100000
ITERATIONS_FOR_STATS = 100    


#NET_FILE_NAME = "NALU_" + str(program_params.number_neurons_1) + "__Dense_" + str(program_params.number_neurons_2)
NET_FILE_NAME = "Dense_" + str(program_params.number_neurons_1) + "__Dense_" + str(program_params.number_neurons_2)
CHECKPOINT_DIR = "./checkpoints_" + NET_FILE_NAME + "/"
CHECKPOINT_NAME = "model"
CHECKPOINT_ITER = 50



data_importer = DataImporter("/home/lidia/projects/Ranker-Net/Drone Dataset/data.csv", seed = RANDOM_SEED_DATA) 
data_importer.load_data()

alphas = [data_importer.data[data_importer.data.object_class==DetectionsLabels.DRONE.value].shape[0] / data_importer.data.shape[0],
            data_importer.data[data_importer.data.object_class!=DetectionsLabels.DRONE.value].shape[0] / data_importer.data.shape[0]]
print("ALPHAS: ", alphas)
model = RankNetwork(INPUT_LEN,NET_STRUCTURE,LEARNING_RATE,alphas)

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
        file_with_statistics = open(NET_FILE_NAME + "_statistics.csv", "w")
        writer_csv = csv.writer(file_with_statistics)

        for X_1,X_2,y in train_batch:
            model.train_step(X_1,X_2)

            step = model.get_global_step()

            if step % CHECKPOINT_ITER == 0:
                X_1,X_2,y = next(test_batch)
                loss, accuracy, loss_median = model.get_metrics(X_1,X_2)
                saver.save(sess,CHECKPOINT_DIR + CHECKPOINT_NAME,global_step=step)
                params = [step, model.get_learning_rate(), loss, loss_median, accuracy]
                writer_csv.writerow(params)

        file_with_statistics.close()
        
        file_with_global_statistics = open("statistics_Dense_Dense.csv", "a")
        writer_csv = csv.writer(file_with_global_statistics)
        loss = 0
        accuracy = 0
        loss_median = 0
        for i in range(ITERATIONS_FOR_STATS):
            X_1,X_2,y = next(test_batch)
            loss_i, accuracy_i, loss_median_i = model.get_metrics(X_1,X_2)
            loss += loss_i
            accuracy += accuracy_i
            loss_median += loss_median_i
        params = [program_params.number_neurons_1, program_params.number_neurons_2, \
                    loss/ITERATIONS_FOR_STATS, loss_median/ITERATIONS_FOR_STATS, accuracy/ITERATIONS_FOR_STATS]
        writer_csv.writerow(params)
        file_with_global_statistics.close()


    except KeyboardInterrupt:
            print("Training interrupted!")
    finally:
        saver.save(sess,CHECKPOINT_DIR + CHECKPOINT_NAME,global_step=model.get_global_step())
        model.dump_network_to_file(NET_FILE_NAME + ".json")
