package com.map.myapplication;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.view.Window;
import android.view.WindowManager;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.ImageView;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    private static int SPLASH_SCREEN=3500;// Duration of splash screen

    //variables
    Animation topAnim,bottomAnim;//Animation objects
    ImageView logo;
    TextView title;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,WindowManager.LayoutParams.FLAG_FULLSCREEN);//To remove the title bar from top //Goto res->values->styles-><style name="AppTheme" parent="Theme.AppCompat.Light.NoActionBar">
        setContentView(R.layout.activity_main);
        topAnim = AnimationUtils.loadAnimation(this,R.anim.top_anim);
        bottomAnim = AnimationUtils.loadAnimation(this,R.anim.bottom_anim);

        logo = findViewById(R.id.logo);
        title = findViewById(R.id.title);

        logo.setAnimation(topAnim);
        title.setAnimation(bottomAnim);

        //Delay Handler
        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                Intent intent=new Intent(MainActivity.this,Dashboard.class);
                startActivity(intent);
                finish();
            }
        },SPLASH_SCREEN);
    }
}