package com.example.edgedetection;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.Image;
import android.media.MediaRecorder;
import android.media.MediaScannerConnection;
import android.os.Build;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v4.content.PermissionChecker;
import android.support.v4.graphics.drawable.DrawableCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import java.nio.ShortBuffer;
import java.util.List;

public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static final String LOG_TAG = "MainActivity";
    final int RECORD_LENGTH = 10;

    private CameraBridgeViewBase mOpenCvCameraView;
    private Mat mRgba;


    long startTime = 0;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(LOG_TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public MainActivity() {
        Log.i(LOG_TAG, "Instantiated new " + this.getClass());
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(LOG_TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        if (checkPermission()) {

        } else {
            ActivityCompat.requestPermissions(MainActivity.this,
                    new String[]{Manifest.permission.CAMERA}, 1234);
        }

        mOpenCvCameraView = (JavaCameraView) findViewById(R.id.camera);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(LOG_TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(LOG_TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }


    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
    }

    public void onCameraViewStopped() {
        if (mRgba != null)
            mRgba.release();

        mRgba = null;
    }

    private MatOfPoint2f mMOP2fptsSafe, mMOP2fptsPrev, mMOP2fptsThis;
    private Mat matOpFlowThis, matOpFlowPrev;
    private MatOfPoint MOPcorners;
    private int iGFFTMax = 500;
    private MatOfByte mMOBStatus;
    private MatOfFloat mMOFerr;
    private List<Byte> byteStatus;


    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        if (mMOP2fptsPrev.rows() == 0) {

            //Log.d("Baz", "First time opflow");
            // first time through the loop so we need prev and this mats
            // plus prev points
            // get this mat
            Imgproc.cvtColor(mRgba, matOpFlowThis, Imgproc.COLOR_RGBA2GRAY);

            // copy that to prev mat
            matOpFlowThis.copyTo(matOpFlowPrev);

            // get prev corners
            Imgproc.goodFeaturesToTrack(matOpFlowPrev, MOPcorners, iGFFTMax, 0.05, 20);
            mMOP2fptsPrev.fromArray(MOPcorners.toArray());

            // get safe copy of this corners
            mMOP2fptsPrev.copyTo(mMOP2fptsSafe);
        } else {
            //Log.d("Baz", "Opflow");
            // we've been through before so
            // this mat is valid. Copy it to prev mat
            matOpFlowThis.copyTo(matOpFlowPrev);

            // get this mat
            Imgproc.cvtColor(mRgba, matOpFlowThis, Imgproc.COLOR_RGBA2GRAY);

            // get the corners for this mat
            Imgproc.goodFeaturesToTrack(matOpFlowThis, MOPcorners, iGFFTMax, 0.05, 20);
            mMOP2fptsThis.fromArray(MOPcorners.toArray());

            // retrieve the corners from the prev mat
            // (saves calculating them again)
            mMOP2fptsSafe.copyTo(mMOP2fptsPrev);

            // and save this corners for next time through

            mMOP2fptsThis.copyTo(mMOP2fptsSafe);
        }


    /*
    Parameters:
        prevImg first 8-bit input image
        nextImg second input image
        prevPts vector of 2D points for which the flow needs to be found; point coordinates must be single-precision floating-point numbers.
        nextPts output vector of 2D points (with single-precision floating-point coordinates) containing the calculated new positions of input features in the second image; when OPTFLOW_USE_INITIAL_FLOW flag is passed, the vector must have the same size as in the input.
        status output status vector (of unsigned chars); each element of the vector is set to 1 if the flow for the corresponding features has been found, otherwise, it is set to 0.
        err output vector of errors; each element of the vector is set to an error for the corresponding feature, type of the error measure can be set in flags parameter; if the flow wasn't found then the error is not defined (use the status parameter to find such cases).
    */
        Video.calcOpticalFlowPyrLK(matOpFlowPrev, matOpFlowThis, mMOP2fptsPrev, mMOP2fptsThis, mMOBStatus, mMOFerr);

        cornersPrev = mMOP2fptsPrev.toList();
        cornersThis = mMOP2fptsThis.toList();

        y = byteStatus.size() - 1;

        for (x = 0; x < y; x++) {
            if (byteStatus.get(x) == 1) {
                Point pt = cornersThis.get(x);
                Point pt2 = cornersPrev.get(x);

                Core.circle(mRgba, pt, 5, colorRed, iLineThickness - 1);

                Core.line(mRgba, pt, pt2, colorRed, iLineThickness);
            }
        }

        return mRgba;

    }

    int frames = 0;

    public Mat stabilizeImage(Mat newFrame, Mat oldFrame, MatOfPoint2f featuresOld) {
        Mat greyNew = new Mat();
        Mat greyOld = new Mat();
        Imgproc.cvtColor(newFrame, greyNew, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(oldFrame, greyOld, Imgproc.COLOR_BGR2GRAY);
        MatOfPoint2f currentFeatures = new MatOfPoint2f();
        MatOfFloat err = new MatOfFloat();
        MatOfByte status = new MatOfByte();
        Video.calcOpticalFlowPyrLK(greyOld, greyNew, featuresOld, currentFeatures, status, err);
        Mat correctionMatrix = Calib3d.estimateAffine2D(currentFeatures, featuresOld);
        Mat corrected = new Mat();
        Imgproc.warpAffine(newFrame, corrected, correctionMatrix, newFrame.size());
        return corrected;
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        switch (requestCode) {
            case 1234: {
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                } else {
                    Toast.makeText(MainActivity.this, "Permission not granted", Toast.LENGTH_SHORT).show();
                    finish();
                }
            }
        }
    }

    private boolean checkPermission() {
        int permissionCheck_Cam = ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.CAMERA);

        if (permissionCheck_Cam == PermissionChecker.PERMISSION_GRANTED) {
            return true;
        } else {
            return false;
        }
    }

}