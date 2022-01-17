import torch, cv2, os, numpy as np
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures.image_list import ImageList
from detectron2.data import transforms as T
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.structures.boxes import Boxes
from detectron2.layers import nms
from detectron2 import model_zoo
from detectron2.config import get_cfg
from transformers import BertTokenizer, VisualBertModel
import torch.nn.functional as F
from models.seqModel import MultiTaskLoss

class VisualBERT(torch.nn.Module):

  def __init__(self,interm_size=64, max_length=120, **kwargs):
    '''
    kwargs min_edge, max_edge, min_boxes, max_boxes
    '''
    super(VisualBERT, self).__init__()


    self.best_acc = None
    self.max_length = max_length
    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.interm_neurons = interm_size

    self.cfg = get_cfg()
    self.cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'))
    self.cfg.INPUT.MAX_SIZE_TEST = kwargs['max_edge']
    self.cfg.INPUT.MIN_SIZE_TEST = kwargs['min_edge']
    
    self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # ROI HEADS SCORE THRESHOLD for taking or not proposed region as background or foreground
    self.cfg['MODEL']['DEVICE']= self.device
    self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml')
    self.FPN = build_model(self.cfg)  #
    checkpointer = DetectionCheckpointer(self.FPN)
    checkpointer.load(self.cfg.MODEL.WEIGHTS)

    self.MIN_BOXES = kwargs['min_boxes']
    self.MAX_BOXES = kwargs['max_boxes']
    


    self.frcnn = FastRCNNOutputLayers(input_shape=1024, box2box_transform = Box2BoxTransform(weights=self.cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
                      num_classes = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES,
                      cls_agnostic_bbox_reg = self.cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
                      smooth_l1_beta = self.cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA, 
                      test_score_thresh = self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
                      test_nms_thresh = self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
                      test_topk_per_image = self.cfg.TEST.DETECTIONS_PER_IMAGE, 
                      box_reg_loss_type = self.cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
                      loss_weight = {"loss_box_reg": self.cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT})
    
    self.VisualBert = VisualBertModel.from_pretrained("uclanlp/visualbert-nlvr2-coco-pre")
    self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    self.intermediate = torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=self.interm_neurons), torch.nn.LeakyReLU())

    if kwargs['multitask'] == True:
      self.classifier = torch.nn.Linear(in_features=self.interm_neurons, out_features=5)
      self.loss_criterion = MultiTaskLoss()
    else: 
      self.classifier = torch.nn.Linear(in_features=self.interm_neurons, out_features=2)
      self.loss_criterion = torch.nn.CrossEntropyLoss()

    
    self.to(device=self.device)

  def image_preprocess(self, img_list):

    # Resizing the image according to the configuration
    transform_gen = T.ResizeShortestEdge(
                   [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
            )
    img_list = [transform_gen.get_transform(img).apply_image(img) for img in img_list]

    # Convert to C,H,W format
    convert_to_tensor = lambda x: torch.Tensor(x.astype("float32").transpose(2, 0, 1))

    batched_inputs = [{"image":convert_to_tensor(img), "height": img.shape[0], "width": img.shape[1]} for img in img_list]

    # Normalizing the image
    num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
    pixel_mean = torch.Tensor(self.cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1)
    pixel_std = torch.Tensor(self.cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1)
    normalizer = lambda x: (x - pixel_mean) / pixel_std
    images = [normalizer(x["image"]) for x in batched_inputs]

    # Convert to ImageList
    images =  ImageList.from_tensors(images, self.FPN.backbone.size_divisibility)
    
    return images, batched_inputs

  def get_box_features(self, features, proposals, batch_size):
    features_list = [features[f] for f in ['p2', 'p3', 'p4', 'p5']]
    box_features = self.FPN.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    box_features = self.FPN.roi_heads.box_head.flatten(box_features)
    box_features = self.FPN.roi_heads.box_head.fc1(box_features)
    box_features = self.FPN.roi_heads.box_head.fc_relu1(box_features)
    box_features = self.FPN.roi_heads.box_head.fc2(box_features)

    box_features = box_features.reshape(batch_size, box_features.shape[0]//batch_size, 1024)
    return box_features, features_list


  def get_ROI_prediction_logits(self, features_list, proposals):
    cls_features = self.FPN.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    cls_features = self.FPN.roi_heads.box_head(cls_features)
    pred_class_logits, pred_proposal_deltas = self.FPN.roi_heads.box_predictor(cls_features)
    
    return pred_class_logits, pred_proposal_deltas


  def get_box_scores(self, pred_class_logits, pred_proposal_deltas, proposals):
  
    boxes = self.frcnn.predict_boxes((pred_class_logits, pred_proposal_deltas), proposals)
    scores = self.frcnn.predict_probs((pred_class_logits, pred_proposal_deltas), proposals)
    image_shapes = image_shapes = [x.image_size for x in proposals]

    return boxes, scores, image_shapes

  def get_output_boxes(slef, boxes, batched_inputs, image_size):
    proposal_boxes = boxes.reshape(-1, 4).clone()
    scale_x, scale_y = (batched_inputs["width"] / image_size[1], batched_inputs["height"] / image_size[0])
    output_boxes = Boxes(proposal_boxes)

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(image_size)

    return output_boxes

  def select_boxes(self, output_boxes, scores):

    test_score_thresh = self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    test_nms_thresh = self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
    cls_prob = scores.detach()
    cls_boxes = output_boxes.tensor.detach()
    cls_boxes = cls_boxes.reshape(len(cls_boxes)//80,80,4)
    max_conf = torch.zeros((cls_boxes.shape[0])).to(self.device)

    for cls_ind in range(0, cls_prob.shape[1]-1):
        cls_scores = cls_prob[:, cls_ind+1]
        det_boxes = cls_boxes[:,cls_ind,:]
        keep = nms(det_boxes, cls_scores, test_nms_thresh)
        max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])
    keep_boxes = torch.where(max_conf >= test_score_thresh)[0]

    return keep_boxes, max_conf

  def filter_boxes(self, keep_boxes, max_conf):

    if len(keep_boxes) < self.MIN_BOXES:
        keep_boxes = np.argsort(max_conf.cpu().numpy())[::-1][:self.MIN_BOXES]
    elif len(keep_boxes) > self.MAX_BOXES:
        keep_boxes = np.argsort(max_conf.cpu().numpy())[::-1][:self.MAX_BOXES]

    return keep_boxes

  
  def get_image_features(self, p):
    

    images, batched_inputs = self.image_preprocess(p)
    
    features = self.FPN.backbone(images.tensor.to(self.device))
    # print('ishape', features.shape)
    proposals, _ = self.FPN.proposal_generator(images, features)
    
    box_features, features_list = self.get_box_features(features, proposals, len(p))
    pred_class_logits, pred_proposal_deltas = self.get_ROI_prediction_logits(features_list, proposals)

    
    boxes, scores, image_shapes = self.get_box_scores(pred_class_logits, pred_proposal_deltas, proposals)
    output_boxes = [self.get_output_boxes(boxes[i], batched_inputs[i], proposals[i].image_size) for i in range(len(proposals))]
    temp = [self.select_boxes(output_boxes[i], scores[i]) for i in range(len(scores))]
    
    keep_boxes, max_conf = [],[]
    for keep_box, mx_conf in temp:
        keep_boxes.append(keep_box)
        max_conf.append(mx_conf)

    keep_boxes = [self.filter_boxes(keep_box, mx_conf) for keep_box, mx_conf in zip(keep_boxes, max_conf)]

    ret = []
    padd_features_boxes = lambda x, t : torch.nn.ZeroPad2d((0, 0, 0, t))(x)
 
    for box_feature, keep_box in zip(box_features, keep_boxes):
      if torch.is_tensor(keep_box):
        keep_box = keep_box.cpu().numpy()
      ret.append(padd_features_boxes(box_feature[keep_box.copy()], self.MAX_BOXES - keep_box.shape[0]))
    
    return ret


  def forward(self, data, get_encoding=False):
    
    self.FPN.eval()
    self.frcnn.eval()
    P = [cv2.cvtColor(cv2.imread(x), cv2.COLOR_RGB2BGR) for x in data['images']]
    visual_embeds = torch.stack([self.get_image_features([p])[0] for p in P]) 

    text = self.tokenizer(data['text'], return_tensors="pt", truncation=True, padding=True, max_length=self.max_length).to(self.device)
    
    visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

    outputs = self.VisualBert(input_ids = text['input_ids'], attention_mask = text['attention_mask'],
              visual_embeds = visual_embeds, visual_token_type_ids = visual_token_type_ids.to(self.device),
              visual_attention_mask = visual_attention_mask.to(self.device), token_type_ids = text['token_type_ids'],
              return_dict = True).pooler_output

    Y = self.intermediate(outputs)

    if get_encoding == True:
      return Y

    return self.classifier(Y)

  def load(self, path):
    self.load_state_dict(torch.load(path, map_location=self.device))

  def save(self, path):
    torch.save(self.state_dict(), path)

  def makeOptimizer(self, lr, decay):
    return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=decay)