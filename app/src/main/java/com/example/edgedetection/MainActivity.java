package com.example.edgedetection;

import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.media.Image;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.ImageView;

import org.opencv.android.Utils;
import org.opencv.core.CvType;

import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class MainActivity extends AppCompatActivity {
    ImageView imageView;
    ImageView detectEdgesImageView;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = findViewById(R.id.imageView);
        detectEdgesImageView = findViewById(R.id.detectEdge);
        BitmapDrawable drawable = (BitmapDrawable) imageView.getDrawable();
        Bitmap bitmap = drawable.getBitmap();
        detectEdges(bitmap);
    }

    private void detectEdges(Bitmap bitmap) {
        Mat rgba = new Mat();
        Utils.bitmapToMat(bitmap, rgba);

        Mat edges = new Mat(rgba.size(), CvType.CV_8UC1);
        Imgproc.cvtColor(rgba, edges, Imgproc.COLOR_RGB2GRAY, 4);
        Imgproc.Canny(edges, edges, 80, 100);

        Bitmap resultBitmap = Bitmap.createBitmap(edges.cols(), edges.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(edges, resultBitmap);
        detectEdgesImageView.setImageBitmap(resultBitmap);
    }
}
