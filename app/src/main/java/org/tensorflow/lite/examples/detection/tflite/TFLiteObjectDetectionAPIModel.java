/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;
import java.util.PriorityQueue;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.detection.env.Logger;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/research/object_detection
 */
public class TFLiteObjectDetectionAPIModel implements Classifier {
  private static final Logger LOGGER = new Logger();

  // Only return this many results.
  private static final int NUM_DETECTIONS = 19;
  // Float model
  private static final float IMAGE_MEAN = 128.0f;
  private static final float IMAGE_STD = 128.0f;
  // Number of threads in the java app
  private static final int NUM_THREADS = 4;
  private boolean isModelQuantized;
  // Config values.
  private int inputSize;
  // Pre-allocated buffers.
  private Vector<String> labels = new Vector<String>();
  private int[] intValues;
  // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
  // contains the location of detected boxes
  private float[][][][] outputLBBox;
  // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the classes of detected boxes
  private float[][][][] outputMBBox;
  // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the scores of detected boxes
  private float[][][][] outputSBBox;
  // numDetections: array of shape [Batchsize]
  // contains the number of detected boxes
  private float[] numDetections;

  private final int blockSize;
  private final double THRESHOLD;
  private final int NUM_CLASSES;
  private final int INP_IMG_WIDTH;
  private final int INP_IMG_HEIGHT;
  private final int gridWidth;
  private final int gridHeight;
  private final int NUM_BOXES_PER_BLOCK;
  private final int MAX_RESULTS;
  private final double OVERLAP_THRESHOLD;
  private List<Map<String, Object>> test_result = new ArrayList<>();
  private final List<Integer> anchors;

  private ByteBuffer imgData;

  private Interpreter tfLite;

  private TFLiteObjectDetectionAPIModel() {
      blockSize = 32;
      THRESHOLD = 0.5;
      NUM_CLASSES = 80;
      INP_IMG_WIDTH = 608;
      INP_IMG_HEIGHT = 608;
      gridWidth = INP_IMG_WIDTH / blockSize;
      gridHeight = INP_IMG_HEIGHT / blockSize;
      NUM_BOXES_PER_BLOCK = 3;
      anchors = new ArrayList<Integer>(){{
          add(10);   add(13);
          add(16);  add(30);
          add(33);  add(23);
          add(30);  add(61);
          add(62);  add(45);
          add(59);  add(119);
          add(116);  add(90);
          add(156);  add(198);
          add(373);  add(326);
      }};
      MAX_RESULTS = 15;
      OVERLAP_THRESHOLD = 0.7;
  }

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
          throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   * @param labelFilename The filepath of label file for classes.
   * @param inputSize The size of image input
   * @param isQuantized Boolean representing model is quantized or not
   */
  public static Classifier create(
          final AssetManager assetManager,
          final String modelFilename,
          final String labelFilename,
          final int inputSize,
          final boolean isQuantized)
          throws IOException {
    final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

    InputStream labelsInput = null;
    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    labelsInput = assetManager.open(actualFilename);
    BufferedReader br = null;
    br = new BufferedReader(new InputStreamReader(labelsInput));
    String line;
    while ((line = br.readLine()) != null) {
      LOGGER.w(line);
      d.labels.add(line);
    }
    br.close();

    d.inputSize = inputSize;

    try {
      d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    d.isModelQuantized = isQuantized;
    // Pre-allocate buffers.
    int numBytesPerChannel;
    if (isQuantized) {
      numBytesPerChannel = 1; // Quantized
    } else {
      numBytesPerChannel = 4; // Floating point
    }
    d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
    d.imgData.order(ByteOrder.nativeOrder());
    d.intValues = new int[d.inputSize * d.inputSize];

    d.tfLite.setNumThreads(NUM_THREADS);
    d.outputLBBox = new float[1][NUM_DETECTIONS][NUM_DETECTIONS][255];
    d.outputMBBox = new float[1][NUM_DETECTIONS * 2][NUM_DETECTIONS * 2][255];
    d.outputSBBox = new float[1][NUM_DETECTIONS * 4][NUM_DETECTIONS * 4][255];
    d.numDetections = new float[1];
    return d;
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    imgData.rewind();
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixelValue = intValues[i * inputSize + j];
        if (isModelQuantized) {
          // Quantized model
          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
          imgData.put((byte) (pixelValue & 0xFF));
        } else { // Float model
          imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        }
      }
    }
    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");
    outputLBBox = new float[1][NUM_DETECTIONS][NUM_DETECTIONS][255];
    outputMBBox = new float[1][NUM_DETECTIONS * 2][NUM_DETECTIONS * 2][255];
    outputSBBox = new float[1][NUM_DETECTIONS * 4][NUM_DETECTIONS * 4][255];
    numDetections = new float[1];

    Object[] inputArray = {imgData};
    Map<Integer, Object> outputMap = new HashMap<>();
    outputMap.put(0, outputLBBox);
    outputMap.put(1, outputMBBox);
    outputMap.put(2, outputSBBox);
    //outputMap.put(3, numDetections);
    Trace.endSection();

    // Run the inference call.
    Trace.beginSection("run");
    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
    Trace.endSection();
    test_result = postProcess(outputLBBox[0], outputMBBox[0], outputSBBox[0]);
    // Show the best detections.
    // after scaling them back to the input size.
    final ArrayList<Recognition> recognitions = new ArrayList<>(NUM_DETECTIONS);
    for (int i = 0; i < test_result.size(); ++i) {
      Map<String, Object> result = test_result.get(i);
      Map<String, Float> primary = (Map<String, Float>) result.get("rect");

      final RectF detection =
              new RectF(
                      primary.get("left"),
                      primary.get("top"),
                      primary.get("right"),
                      primary.get("bottom"));
      // SSD Mobilenet V1 Model assumes class 0 is background class
      // in label file and class labels start from 0 to number_of_classes,
      // while outputClasses correspond to class index from 0 to number_of_classes
      recognitions.add(
              new Recognition(
                      "" + i,
                      labels.get((int) result.get("classIndex")),
                      (float)result.get("confidence"),
                      detection));
    }
    Trace.endSection(); // "recognizeImage"
    return recognitions;
  }

  public List<Map<String, Object>> postProcess(final float[][][] outLBBox, final float[][][] outMBBox, final float[][][] outSBBox ) {
    // Find the best detections.
    PriorityQueue<Map<String, Object>> priorityQueue = new PriorityQueue<>(1, new PredictionComparator());
    //float[][][] test_map = (float[][][]) outputMap.get(0);
    float xPos = 0;
    float yPos = 0;

    float w = 0;
    float h = 0;
    float confidenceInClass = 0;
    float[] classes;
    int offset = 0;
    float confidence = 0;
    for(int z = 0; z < 3; ++z){
      for(int y = 0; y < gridHeight * (int)Math.pow((double)2,(double)z); ++y){
        for(int x = 0; x < gridWidth * (int)Math.pow((double)2,(double)z); ++x ){
          for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
            offset = (NUM_CLASSES + 5) * b;
            if(z == 0){
              confidence = expit(outLBBox[y][x][offset + 4]);
            }else if(z == 1){
              confidence = expit(outMBBox[y][x][offset + 4]);
            }else if(z == 2){
              confidence = expit(outSBBox[y][x][offset + 4]);
            }

            int detectedClass = -1;
            float maxClass = 0;

            classes = new float[NUM_CLASSES];
            for (int c = 0; c < NUM_CLASSES; ++c) {
              if(z == 0){
                classes[c] = outLBBox[y][x][offset + 5 + c];
              }else if(z == 1){
                classes[c] = outMBBox[y][x][offset + 5 + c];
              }else if(z == 2){
                classes[c] = outSBBox[y][x][offset + 5 + c];
              }
            }
            softmax(classes);

            for (int c = 0; c < NUM_CLASSES; ++c) {
              if (classes[c] > maxClass) {
                detectedClass = c;
                maxClass = classes[c];
              }
            }

            confidenceInClass = maxClass * confidence;
//          final float confidenceInClass = maxClass;
            if (confidenceInClass >  THRESHOLD) {
              Map<String, Object> prediction = new HashMap<>();
              prediction.put("classIndex",detectedClass);
              prediction.put("confidence",confidenceInClass);
              if(z == 0){
                float a_1 = expit(outLBBox[y][x][offset + 0]);
                float a_2 = expit(outLBBox[y][x][offset + 1]);
                float a_3 = (float)(Math.exp(outLBBox[y][x][offset + 2]));
                float a_4 = (float)(Math.exp(outLBBox[y][x][offset + 3]));
                xPos = (x + expit(outLBBox[y][x][offset + 0])) * blockSize;
                yPos = (y + expit(outLBBox[y][x][offset + 1])) * blockSize;
                w = (float) (Math.exp(outLBBox[y][x][offset + 2]) * anchors.get(2 * (b + 3 * z) + 0)) * blockSize;
                h = (float) (Math.exp(outLBBox[y][x][offset + 3]) * anchors.get(2 * (b + 3 * z) + 1)) * blockSize;
              }else if(z == 1){
                xPos = (x + expit(outMBBox[y][x][offset + 0])) * 16;
                yPos = (y + expit(outMBBox[y][x][offset + 1])) * 16;
                w = (float) (Math.exp(outMBBox[y][x][offset + 2]) * anchors.get(2 * (b + 3 * z) + 0)) * 16;
                h = (float) (Math.exp(outMBBox[y][x][offset + 3]) * anchors.get(2 * (b + 3 * z) + 1)) * 16;
              }else if(z == 2){
                xPos = (x + expit(outSBBox[y][x][offset + 0])) * 8;
                yPos = (y + expit(outSBBox[y][x][offset + 1])) * 8;
                w = (float) (Math.exp(outSBBox[y][x][offset + 2]) * anchors.get(2 * (b + 3 * z) + 0)) * 8;
                h = (float) (Math.exp(outSBBox[y][x][offset + 3]) * anchors.get(2 * (b + 3 * z) + 1)) * 8;
              }
              Map<String, Float> rectF = new HashMap<>();
              rectF.put("left", Math.max(0, xPos - w / 2)); // left should have lower value as right
              rectF.put("top", Math.max(0, yPos - h / 2));  // top should have lower value as bottom
              rectF.put("right",Math.min(INP_IMG_WIDTH- 1, xPos + w / 2));
              rectF.put("bottom",Math.min(INP_IMG_HEIGHT - 1, yPos + h / 2));
              rectF.put("centerXPos", xPos);
              rectF.put("centerYPos", yPos);
              prediction.put("rect",rectF);
              priorityQueue.add(prediction);
            }
          }
        }
      }
    }

    final List<Map<String, Object>> predictions = new ArrayList<>();
    Map<String, Object> bestPrediction = priorityQueue.poll();
    predictions.add(bestPrediction);

   // for (int i = 0; i < Math.min(priorityQueue.size(), MAX_RESULTS); ++i) {
    for (int i = 0; i < priorityQueue.size(); ++i) {
      Map<String, Object> prediction = priorityQueue.poll();
      boolean overlaps = false;
      for (Map<String, Object> previousPrediction : predictions) {
        float intersectProportion = 0f;
        Map<String, Float> primary = (Map<String, Float>) previousPrediction.get("rect");
        Map<String, Float> secondary =  (Map<String, Float>) prediction.get("rect");
        if (primary.get("left") < secondary.get("right") && primary.get("right") > secondary.get("left")
                && primary.get("top") < secondary.get("bottom") && primary.get("bottom") > secondary.get("top")) {
          float intersection = Math.max(0, Math.min(primary.get("right"), secondary.get("right")) - Math.max(primary.get("left"), secondary.get("left"))) *
                  Math.max(0, Math.min(primary.get("bottom"), secondary.get("bottom")) - Math.max(primary.get("top"), secondary.get("top")));

          float main = Math.abs(primary.get("right") - primary.get("left")) * Math.abs(primary.get("bottom") - primary.get("top"));
          intersectProportion= intersection / main;
        }

        overlaps = overlaps || (intersectProportion > OVERLAP_THRESHOLD);
      }

      if (!overlaps) {
        predictions.add(prediction);
      }
    }
    return predictions;
  }

  private class PredictionComparator implements Comparator<Map<String, Object>> {
    @Override
    public int compare(final Map<String, Object> prediction1, final Map<String, Object> prediction2) {
      return Float.compare((float)prediction2.get("confidence"), (float)prediction1.get("confidence"));
    }
  }

  private float expit(final float x) {
    return (float) (1. / (1. + Math.exp(-x)));
  }

  private void softmax(final float[] vals) {
    float max = Float.NEGATIVE_INFINITY;
    for (final float val : vals) {
      max = Math.max(max, val);
    }
    float sum = 0.0f;
    for (int i = 0; i < vals.length; ++i) {
      vals[i] = (float) Math.exp(vals[i] - max);
      sum += vals[i];
    }
    for (int i = 0; i < vals.length; ++i) {
      vals[i] = vals[i] / sum;
    }
  }

  @Override
  public void enableStatLogging(final boolean logStats) {}

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {}

  public void setNumThreads(int num_threads) {
    if (tfLite != null) tfLite.setNumThreads(num_threads);
  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
    if (tfLite != null) tfLite.setUseNNAPI(isChecked);
  }
}
