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

package org.tensorflow.lite.examples.detection.tracking;

import android.content.Context;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Cap;
import android.graphics.Paint.Join;
import android.graphics.Paint.Style;
import android.graphics.Rect;
import android.graphics.RectF;
import android.text.TextUtils;
import android.text.format.Time;
import android.util.Pair;
import android.util.TypedValue;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Random;
import java.util.Timer;

import org.tensorflow.lite.examples.detection.R;
import org.tensorflow.lite.examples.detection.env.BorderedText;

import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Detector.Recognition;

/** A tracker that handles non-max suppression and matches existing objects to new detections. */
public class MultiBoxTracker {
  private static final float TEXT_SIZE_DIP = 18;
  private static final float MIN_SIZE = 16.0f;
  private static final int[] COLORS = {
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.CYAN,
    Color.MAGENTA,
    Color.WHITE,
    Color.parseColor("#55FF55"),
    Color.parseColor("#FFA500"),
    Color.parseColor("#FF8888"),
    Color.parseColor("#AAAAFF"),
    Color.parseColor("#FFFFAA"),
    Color.parseColor("#55AAAA"),
    Color.parseColor("#AA33AA"),
    Color.parseColor("#0D0068")
  };
  final List<Pair<Float, RectF>> screenRects = new LinkedList<Pair<Float, RectF>>();
  private final Logger logger = new Logger();
  private final Queue<Integer> availableColors = new LinkedList<Integer>();
  private final List<TrackedRecognition> trackedObjects = new LinkedList<TrackedRecognition>();
  private final Paint boxPaint = new Paint();

  private final float textSizePx;
  private final BorderedText borderedText;

  private Matrix frameToCanvasMatrix;
  private int frameWidth;
  private int frameHeight;
  private int sensorOrientation;
  private Context context;

  private String title = "gay";
  private final int maxVirus = 20;
  private final int[][] positions = new int[maxVirus][2];
  private final int[] rectPosition = new int[2];

  public MultiBoxTracker(final Context context) {
    this.context = context;

    for (final int color : COLORS) {
      availableColors.add(color);
    }

    boxPaint.setColor(Color.RED);
    boxPaint.setStyle(Style.STROKE);
    boxPaint.setStrokeWidth(10.0f);
    boxPaint.setStrokeCap(Cap.ROUND);
    boxPaint.setStrokeJoin(Join.ROUND);
    boxPaint.setStrokeMiter(100);


    textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, context.getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
  }


  public synchronized void setFrameConfiguration(
      final int width, final int height, final int sensorOrientation) {
    frameWidth = width;
    frameHeight = height;
    this.sensorOrientation = sensorOrientation;
  }

  public synchronized void drawDebug(final Canvas canvas) {
    final Paint textPaint = new Paint();
    textPaint.setColor(Color.WHITE);
    textPaint.setTextSize(60.0f);

    final Paint boxPaint = new Paint();
    boxPaint.setColor(Color.RED);
    boxPaint.setAlpha(200);
    boxPaint.setStyle(Style.STROKE);

    for (final Pair<Float, RectF> detection : screenRects) {
      final RectF rect = detection.second;
      canvas.drawRect(rect, boxPaint);
      canvas.drawText("" + detection.first, rect.left, rect.top, textPaint);

      borderedText.drawText(canvas, rect.centerX(), rect.centerY(), "" + detection.first);
    }
  }

  public synchronized void trackResults(final List<Recognition> results, final long timestamp) {
    logger.i("Processing %d results from %d", results.size(), timestamp);
    processResults(results);
  }

  private Matrix getFrameToCanvasMatrix() {
    return frameToCanvasMatrix;
  }

  public synchronized void draw(final Canvas canvas) {

    final boolean rotated = sensorOrientation % 180 == 90;
    final float multiplier =
        Math.min(
            canvas.getHeight() / (float) (rotated ? frameWidth : frameHeight),
            canvas.getWidth() / (float) (rotated ? frameHeight : frameWidth));
    frameToCanvasMatrix =
        ImageUtils.getTransformationMatrix(
            frameWidth,
            frameHeight,
            (int) (multiplier * (rotated ? frameHeight : frameWidth)),
            (int) (multiplier * (rotated ? frameWidth : frameHeight)),
            sensorOrientation,
            false);

    if (trackedObjects.size() != 0) {
      final TrackedRecognition recognition = trackedObjects.get(0);
      final RectF trackedPos = new RectF(recognition.location);

      getFrameToCanvasMatrix().mapRect(trackedPos);
      boxPaint.setColor(recognition.color);

      float cornerSize = Math.min(trackedPos.width(), trackedPos.height()) / 8.0f;
      canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint);

      Paint p = new Paint();
      Bitmap b = BitmapFactory.decodeResource(context.getResources(), R.drawable.virus_144px);
      b = getResizedBitmap(b, 100, 100);
      Thread t = new Thread(() -> {
        if (!recognition.title.equals(title)) {
          title = recognition.title;
          rectPosition[0] = (int) Math.floor((double) trackedPos.left);
          rectPosition[1] = (int) Math.floor((double) trackedPos.top);

          for (int i=0 ; i<maxVirus ; i++) {

            int x = getRandomNumberInRange(
                    (int) Math.floor((double) trackedPos.left),
                    (int) Math.floor((double) trackedPos.right));
            int y = getRandomNumberInRange(
                    (int) Math.floor((double) trackedPos.top),
                    (int) Math.floor((double) trackedPos.bottom));

            positions[i][0] = x;
            positions[i][1] = y;
          }

        } else {
          for (int i=0 ; i<maxVirus ; i++) {
            positions[i][0] = (int)trackedPos.left-rectPosition[0]+positions[i][0];
            positions[i][1] = (int)trackedPos.top-rectPosition[1]+positions[i][1];

          }
          rectPosition[0] = (int) Math.floor((double) trackedPos.left);
          rectPosition[1] = (int) Math.floor((double) trackedPos.top);
        }

      });
      try {
        t.start();
        t.join();
      } catch (InterruptedException e) {
        e.printStackTrace();
      }

      int counter = 0;
      List<int[]> list = new ArrayList<>();
      Collections.addAll(list, positions);
      int virusSize = 72;
      for (int i=0 ; i<maxVirus ; i++) {
        Rect r1 = new Rect(
                positions[i][0],
                positions[i][1],
                positions[i][0]+virusSize,
                positions[i][1]+virusSize);
        for (int j=0 ; j<maxVirus ; j++) {
          Rect r2 = new Rect(
                  positions[j][0],
                  positions[j][1],
                  positions[j][0]+virusSize,
                  positions[j][1]+virusSize);
          if (r1.centerX() != r2.centerX() && r1.centerY() != r2.centerY()) {
            if (Rect.intersects(r1,r2)) {
              list.remove(i-counter);
              counter++;
              break;
            }
          }
        }
      }

      for (int[] position : list) {
        canvas.drawBitmap(b,position[0]-52,position[1]-52,p);
      }

     /* TODO : MOTS + POURCENTAGE : A SUPPRIMER
      final String labelString =
          !TextUtils.isEmpty(recognition.title)
              ? String.format("%s %.2f", recognition.title, (100 * recognition.detectionConfidence))
              : String.format("%.2f", (100 * recognition.detectionConfidence));
      //            borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.top,
      // labelString);

      borderedText.drawText(
          canvas, trackedPos.left + cornerSize, trackedPos.top, labelString + "%", boxPaint);*/
    }
  }

  private void processResults(final List<Recognition> results) {
    final List<Pair<Float, Recognition>> rectsToTrack = new LinkedList<Pair<Float, Recognition>>();

    screenRects.clear();
    final Matrix rgbFrameToScreen = new Matrix(getFrameToCanvasMatrix());

    for (final Recognition result : results) {
      if (result.getLocation() == null) {
        continue;
      }
      final RectF detectionFrameRect = new RectF(result.getLocation());

      final RectF detectionScreenRect = new RectF();
      rgbFrameToScreen.mapRect(detectionScreenRect, detectionFrameRect);

      logger.v(
          "Result! Frame: " + result.getLocation() + " mapped to screen:" + detectionScreenRect);

      screenRects.add(new Pair<Float, RectF>(result.getConfidence(), detectionScreenRect));

      if (detectionFrameRect.width() < MIN_SIZE || detectionFrameRect.height() < MIN_SIZE) {
        logger.w("Degenerate rectangle! " + detectionFrameRect);
        continue;
      }

      rectsToTrack.add(new Pair<Float, Recognition>(result.getConfidence(), result));
    }

    trackedObjects.clear();
    if (rectsToTrack.isEmpty()) {
      logger.v("Nothing to track, aborting.");
      return;
    }

    for (final Pair<Float, Recognition> potential : rectsToTrack) {
      final TrackedRecognition trackedRecognition = new TrackedRecognition();
      trackedRecognition.detectionConfidence = potential.first;
      trackedRecognition.location = new RectF(potential.second.getLocation());
      trackedRecognition.title = potential.second.getTitle();
      trackedRecognition.color = COLORS[trackedObjects.size()];
      trackedObjects.add(trackedRecognition);

      if (trackedObjects.size() >= COLORS.length) {
        break;
      }
    }
  }

  private static class TrackedRecognition {
    RectF location;
    float detectionConfidence;
    int color;
    String title;
  }

  private Bitmap getResizedBitmap(Bitmap bm, int newHeight, int newWidth) {
    int width = bm.getWidth();
    int height = bm.getHeight();
    float scaleWidth = ((float) newWidth) / width;
    float scaleHeight = ((float) newHeight) / height;
    // create a matrix for the manipulation
    Matrix matrix = new Matrix();
    // resize the bit map
    matrix.postScale(scaleWidth, scaleHeight);
    // recreate the new Bitmap
    return Bitmap.createBitmap(bm, 0, 0, width, height, matrix, false);
  }

  private Bitmap RotateBitmap(Bitmap source, float angle) {
    Matrix matrix = new Matrix();
    matrix.postRotate(angle);
    return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
  }

  private static int getRandomNumberInRange(int min, int max) {

    if (min >= max) {
      throw new IllegalArgumentException("max must be greater than min");
    }

    Random r = new Random();
    return r.nextInt((max - min) + 1) + min;
  }
}
