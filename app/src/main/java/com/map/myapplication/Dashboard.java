package com.map.myapplication;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.view.View;
import android.view.WindowManager;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

import static android.Manifest.permission.ACCESS_FINE_LOCATION;
import static android.Manifest.permission.CAMERA;
import static android.Manifest.permission.READ_EXTERNAL_STORAGE;
import static android.Manifest.permission.WRITE_EXTERNAL_STORAGE;

public class Dashboard extends AppCompatActivity {

    private static final int IMAGE_REQUEST=1;

    String currentImagePath=null;

    Button displayBtn;
    ImageView help,capture,cal;
    TextView tv;
    private static final int PERMISSION_REQUEST_CODE = 200;
    Animation topAnim;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_dashboard);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,WindowManager.LayoutParams.FLAG_FULLSCREEN);
        topAnim= AnimationUtils.loadAnimation(this,R.anim.top_anim);
        tv=findViewById(R.id.textView);
        tv.setAnimation(topAnim);
        if (!checkPermission()) {

            requestPermission();

        }


        cal=findViewById(R.id.cal);
        cal.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent idiot = new Intent(Dashboard.this,Calc.class);
                startActivity(idiot);
            }
        });
        displayBtn=findViewById(R.id.disbtn);
        displayBtn.setVisibility(View.INVISIBLE);
        displayBtn.setActivated(false);
        displayBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(currentImagePath != null && !currentImagePath.isEmpty()) {
                    Toast.makeText(Dashboard.this, currentImagePath, Toast.LENGTH_SHORT).show();
                    Intent intent = new Intent(Dashboard.this, Result.class);
                    intent.putExtra("image_paths", currentImagePath);
                    startActivity(intent);
                }
                else{
                    Toast.makeText(Dashboard.this, "No image captured", Toast.LENGTH_SHORT).show();
                }
            }
        });
        help=findViewById(R.id.help);
        help.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent=new Intent(Dashboard.this,HowToUse.class);
                startActivity(intent);
            }
        });
        capture=findViewById(R.id.capture);
        capture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                int id = view.getId();




                Intent cameraIntent=new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

                if(cameraIntent.resolveActivity(getPackageManager())!=null)
                {
                    File imageFile=null;
                    try {
                        imageFile=getImageFile();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    if(imageFile!=null)
                    {
                        Uri imageUri=FileProvider.getUriForFile(Dashboard.this,"com.map.myapplication.fileprovider",imageFile);
                        cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT,imageUri);
                        startActivityForResult(cameraIntent,IMAGE_REQUEST);
                    }
                }
            }
        });


    }
    private boolean checkPermission() {
        int result = ContextCompat.checkSelfPermission(getApplicationContext(), WRITE_EXTERNAL_STORAGE);

        int result1 = ContextCompat.checkSelfPermission(getApplicationContext(), CAMERA);
        int result2 = ContextCompat.checkSelfPermission(getApplicationContext(), READ_EXTERNAL_STORAGE);

        return result == PackageManager.PERMISSION_GRANTED && result1 == PackageManager.PERMISSION_GRANTED && result1 == PackageManager.PERMISSION_GRANTED;
    }

    private void requestPermission() {

        ActivityCompat.requestPermissions(this, new String[]{WRITE_EXTERNAL_STORAGE, CAMERA,READ_EXTERNAL_STORAGE}, PERMISSION_REQUEST_CODE);

    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String permissions[], int[] grantResults) {
        switch (requestCode) {
            case PERMISSION_REQUEST_CODE:
                if (grantResults.length > 0) {

                    boolean locationAccepted = grantResults[0] == PackageManager.PERMISSION_GRANTED;
                    boolean cameraAccepted = grantResults[1] == PackageManager.PERMISSION_GRANTED;



                       // Snackbar.make(view, "Permission Denied, You cannot access location data and camera.", Snackbar.LENGTH_LONG).show();

                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                            if (shouldShowRequestPermissionRationale(WRITE_EXTERNAL_STORAGE)) {
                                showMessageOKCancel("You need to allow access to both the permissions",
                                        new DialogInterface.OnClickListener() {
                                            @Override
                                            public void onClick(DialogInterface dialog, int which) {
                                                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                                                    requestPermissions(new String[]{WRITE_EXTERNAL_STORAGE, CAMERA,READ_EXTERNAL_STORAGE},
                                                            PERMISSION_REQUEST_CODE);
                                                }
                                            }
                                        });
                                return;
                            }
                        }


                }


                break;
        }
    }


    private void showMessageOKCancel(String message, DialogInterface.OnClickListener okListener) {
        new AlertDialog.Builder(Dashboard.this)
                .setMessage(message)
                .setPositiveButton("OK", okListener)
                .setNegativeButton("Cancel", null)
                .create()
                .show();
    }


    private File getImageFile() throws IOException
    {
        String timeStamp=new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageName="jpg_"+timeStamp+"_";
        File storageDir=getExternalFilesDir(Environment.DIRECTORY_PICTURES);

        File imageFile=File.createTempFile(imageName,".jpg",storageDir);
        currentImagePath=imageFile.getAbsolutePath();
        return imageFile;

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(resultCode==RESULT_CANCELED){
            currentImagePath=null;
        }
        else if(resultCode==RESULT_OK){
            displayBtn.setActivated(true);
            displayBtn.setVisibility(View.VISIBLE);
        }
    }

    /*@Override
    protected void onResume() {
        super.onResume();
        displayBtn.setActivated(false);
        displayBtn.setVisibility(View.INVISIBLE);
    }*/

    /*@Override
    protected void onRestart() {
        super.onRestart();
        displayBtn.setActivated(false);
        displayBtn.setVisibility(View.INVISIBLE);
    }*/

    @Override
    protected void onPause() {
        super.onPause();
        displayBtn.setActivated(false);
        displayBtn.setVisibility(View.INVISIBLE);
    }

 /*   @Override
    protected void onStop() {
        super.onStop();
        displayBtn.setActivated(false);
        displayBtn.setVisibility(View.INVISIBLE);
    }*/
}