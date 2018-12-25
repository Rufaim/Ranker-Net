import numpy as np
import pandas as pd
import enum


class DetectionsLabels(enum.Enum):
    DRONE = 'DRONE'
    OTHER = 'OTHER'
    BULLSHIT = 'BULLSHIT'



class DataImporter(object):
    class_ranks = {DetectionsLabels.DRONE: 2, DetectionsLabels.OTHER: 1, DetectionsLabels.BULLSHIT: 0}
    combs = [(DetectionsLabels.DRONE, DetectionsLabels.OTHER),
            (DetectionsLabels.DRONE, DetectionsLabels.BULLSHIT)
            ]

    feature_columns = ["speed_stability",
                    "estimated_coverage",
                    "size_mean_orthogonal_gradient",
                    "estimated_speed",
                    "estimated_Mahalanobis_distance",
                    "uavity",
                    "speed_stability_ratio",
                    "speed_direction_stability",
                    "speed_atan2",
                    "acceleration_atan2",
                    "mass_centre_x",
                    "mass_centre_y",
                    "bbox_width",
                    "bbox_height",
                    "speed_stability_std",
                    "estimated_coverage_std",
                    "size_mean_orthogonal_gradient_std",
                    "estimated_speed_std",
                    "estimated_Mahalanobis_distance_std",
                    "uavity_std",
                    "speed_stability_ratio_std",
                    "speed_direction_stability_std",
                    "speed_atan2_std",
                    "acceleration_atan2_std",
                    "mass_centre_x_std",
                    "mass_centre_y_std",
                    "bbox_width_std",
                    "bbox_height_std"
                        ]

    def __init__(self,filename,train_test_split_portion=0.30,seed=None):
        self.filename = filename
        self.train_test_split_portion = train_test_split_portion
        self.state = np.random.RandomState(seed)

    def load_data(self):
        self.data = pd.read_csv(self.filename,index_col="Unnamed: 0")
        self.data = self.data[self.data["is_zoom_request"] == 1]
        self.data = self.data.dropna()
        self.data = self.data.reset_index()
        self._postprocess()
    
    def _postprocess(self):
        self.data_per_label_train = {}
        self.data_per_label_test = {}
        for cat in DataImporter.class_ranks.keys():
            t = self.data[self.data["object_class"] == cat.value ]
            t = t[DataImporter.feature_columns].values.astype(np.float)
            test_bool = np.random.random(size=t.shape[0]) < self.train_test_split_portion
            self.data_per_label_train[cat] = t[test_bool]
            self.data_per_label_test[cat] = t[np.logical_not(test_bool)]

    def random_batch_benerator(self, batch_size, num_outs):
        return self._random_batch_generator(self.data_per_label_train, batch_size, num_outs), \
                self._random_batch_generator(self.data_per_label_test, batch_size, num_outs)

    def _random_batch_generator(self, data_per_label, batch_size, num_outs):
        def get_rank(cat_1,cat_2):
            if cat_1 > cat_2:
                return 0
            if cat_1 == cat_2:
                return 1
            if cat_1 < cat_2:
                return 2
            return -10

        L = batch_size//(2*len(DataImporter.combs))
        L = int(np.sqrt(L))
        for i in range(num_outs):
            batch_x1 = []
            batch_x2 = []
            batch_y = []

            idxs = {}
            for cat in DataImporter.class_ranks.keys():
                idxs[cat] = self.state.randint(0,len(data_per_label[cat]),size=L)

            for cat_1, cat_2 in DataImporter.combs:
                t_idx = np.array(np.meshgrid(idxs[cat_1], idxs[cat_2])).T.reshape((-1,2))

                x1 = data_per_label[cat_1][np.reshape(t_idx[:,0],(-1,))]
                x2 = data_per_label[cat_2][np.reshape(t_idx[:,1],(-1,))]

                x1 = np.abs(x1)
                x2 = np.abs(x2)

                batch_x1.append(x1)
                batch_x2.append(x2)
                batch_y.append(np.ones((L**2,))*get_rank(DataImporter.class_ranks[cat_1],DataImporter.class_ranks[cat_2]))

            batch_x1 = np.concatenate(batch_x1,axis=0)
            batch_x2 = np.concatenate(batch_x2,axis=0)
            batch_y = np.concatenate(batch_y,axis=0).astype(np.int)

            yield batch_x1, batch_x2 , np.eye(3)[batch_y]
