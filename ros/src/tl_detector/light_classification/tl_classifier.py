import numpy as np
import rospy
import tensorflow as tf
import yaml
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        SIM_MODEL_PATH = 'sim/frozen_inference_graph.pb'
        REAL_MODEL_PATH = 'real/frozen_inference_graph.pb'
        config_string = rospy.get_param("/traffic_light_config")
        config = yaml.load(config_string)
        MODEL_PATH = REAL_MODEL_PATH if config['is_site'] else SIM_MODEL_PATH
        rospy.logwarn("Chosen Model: {0}".format(MODEL_PATH))
        
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()            
            with tf.gfile.GFile(MODEL_PATH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.sess = tf.Session(graph=self.detection_graph)
    
    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(image, axis=0)  
            (boxes, scores, classes) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes],
                feed_dict={self.image_tensor: img_expanded})
            
            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            # Filter boxes with a confidence score less than `confidence_cutoff`
            confidence_cutoff = 0.8
            filtered_boxes, filtered_scores, filtered_classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)

        # Make light Unknown by default.
        top_class = 4
        if (filtered_classes.any()):
            # Top class from high confidence inference.
            top_class = filtered_classes[0].item()
        elif classes.any():
            # Make best guess from aggregation of lower confidence inference classes.
            confidence_cutoff = 0.4
            filtered_boxes, filtered_scores, filtered_classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)
            class_count = [0,0,0,0]
            for classification in filtered_classes:
                if classification.item() == 1.:
                    class_count[0] += 1
                elif classification.item() == 2.:
                    class_count[1] += 1
                elif classification.item() == 3.:
                    class_count[2] += 1
                else:
                    class_count[3] += 1
            max_count = max(class_count)
            top_class = [i for i, count in enumerate(class_count) if count == max_count][-1] + 1
        else:
            return TrafficLight.UNKNOWN
        
        if top_class == 1.:
            return TrafficLight.GREEN
        elif top_class == 2.:
            return TrafficLight.RED
        elif top_class == 3.:
            return TrafficLight.YELLOW
        else:
            return TrafficLight.UNKNOWN