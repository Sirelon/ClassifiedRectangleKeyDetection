package com.mokshithvoodarla.classifiedrectanglekeydetection;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.view.SurfaceView;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity implements CvCameraViewListener2 {
    private CameraBridgeViewBase mOpenCvCameraView;
    private Mat blackAndWhite;
    private Mat downRes;
    private Mat upRes;
    private Mat contourImg;
    private Mat hoVector;
    private MatOfPoint2f specialzedRect;
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.java_camera_view);
        mOpenCvCameraView.setMaxFrameSize(640, 480);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            String permission = Manifest.permission.CAMERA;
            int requestCode = 0x5;
            if (ContextCompat.checkSelfPermission(MainActivity.this, permission) != PackageManager.PERMISSION_GRANTED) {
                if (ActivityCompat.shouldShowRequestPermissionRationale(MainActivity.this, permission)) {
                    ActivityCompat.requestPermissions(MainActivity.this, new String[]{permission}, requestCode);
                } else {
                    ActivityCompat.requestPermissions(MainActivity.this, new String[]{permission}, requestCode);
                }
            }
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    blackAndWhite = new Mat();
                    downRes = new Mat();
                    upRes = new Mat();
                    contourImg = new Mat();
                    hoVector = new Mat();
                    specialzedRect = new MatOfPoint2f();
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
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        Mat gryImg = inputFrame.gray();
        Mat sameImg = inputFrame.rgba();
        Imgproc.pyrDown(gryImg, downRes, new Size(gryImg.cols()/2, gryImg.rows()/2));
        Imgproc.pyrUp(downRes, upRes, gryImg.size());
        Imgproc.dilate(upRes, upRes, new Mat(), new Point(-1, 1), 1);
        Imgproc.Canny(upRes, blackAndWhite, 50, 200);
        Imgproc.dilate(blackAndWhite, blackAndWhite, new Mat(), new Point(-1, 1), 1);
        List<MatOfPoint> contours = new ArrayList<>();
        contourImg = blackAndWhite.clone();
        Imgproc.findContours(contourImg, contours, hoVector, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        for (MatOfPoint contour : contours) {
            MatOfPoint2f crv = new MatOfPoint2f(contour.toArray());
            Imgproc.approxPolyDP(crv, specialzedRect, 0.02 * Imgproc.arcLength(crv, true), true);
            double carea = Imgproc.contourArea(contour);
            if (Math.abs(carea)<550) {
                continue;
            }
            setLabel(sameImg, "<>", contour);
        }
        return sameImg;
    }
    private void setLabel(Mat im, String label, MatOfPoint contour) {
        int[] bl = new int[1];
        int theFOnt = Core.FONT_HERSHEY_DUPLEX;
        Size chars = Imgproc.getTextSize(label, theFOnt, 1, 3, bl);
        Rect rectang = Imgproc.boundingRect(contour);
        Point point = new Point(rectang.x + ((rectang.width - chars.width) / 2), rectang.y + ((rectang.height + chars.height) /2));
        Imgproc.putText(im, label, point, theFOnt, 1, new Scalar(0,0,255), 3);


    }

    @Override
    public void onPause(){
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }


    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallback);
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }


    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }
    public void onCameraViewStarted(int width, int height) {
    }
    public void onCameraViewStopped() {
    }

}
