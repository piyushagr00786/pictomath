package com.map.myapplication;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Base64;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.ByteArrayOutputStream;

public class Result extends AppCompatActivity {

    ImageView dImage;
    TextView t1;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);
        dImage=findViewById(R.id.displayimage);
        t1=findViewById(R.id.textView2);
        if(!Python.isStarted()){
            Python.start(new AndroidPlatform(this));
        }

        Bitmap bitmap= BitmapFactory.decodeFile(getIntent().getStringExtra("image_paths"));
        dImage.setImageBitmap(bitmap);
        ByteArrayOutputStream b2=new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG,100,b2);
        byte b1[]= b2.toByteArray();


        String img1=android.util.Base64.encodeToString(b1, Base64.DEFAULT);
        Python python = Python.getInstance();
        PyObject pythonFile = python.getModule("predict");
        PyObject b3=pythonFile.callAttr("predict",img1);
        String se=b3.toString();
        t1.setText(se);



    }
}