
/**
 * [html: <canvas id="pie1"></canvas>]
 * [js: new Chart("pie1").ratePie(60);]
 * author:zhengyong
 * create:2016-05-11
 */
  window.Chart=function(domId){
    var canvas = document.getElementById(domId);
    var ctx = canvas.getContext("2d");
    var W = canvas.width;
    var H = canvas.height;
    var deg=0,new_deg=0,dif=0;
    var text,text_w;
    var R=W<H?W:H;//先默认环半径为canvas宽度
    var bgR=(R-R/3)/2;//内环背景半径
    var linkW=R/2-bgR;//环宽度=(W/2-内半径)
    var inW=linkW/3;//内环宽度=环宽度*1/3
    var outW=linkW*2/3;//外环宽度=环宽度*2/3
    var inR=bgR+inW/2;//内环半径=内半径+内环宽度/2
    var outR=bgR+inW+outW/2;//外环半径=内半径+内环宽度+外环宽度/2
    var $this=this;
    $this.ratePieNoAnimation=function(number){
        deg=360*number/100;
          //底色大
        ctx.clearRect(0,0,W,H);
        ctx.beginPath();
        var grd = ctx.createLinearGradient(0,H, W, 0);//颜色渐变的起始坐标和终点坐标
            grd.addColorStop(0, "#00b4b2");
            grd.addColorStop(0.3, "#009691");
            grd.addColorStop(0.6, "#00bab8");
            grd.addColorStop(1, "#007d7e");
            ctx.strokeStyle = grd;//生成的颜色块赋值给样式
        ctx.lineWidth=outW;
        ctx.arc(W/2,H/2,outR,0,Math.PI*2,false);
        ctx.stroke();

        //底色小
        ctx.beginPath();
        var grd = ctx.createLinearGradient(W, 0, 0,H);
           grd.addColorStop(0, "#9cded8");
            grd.addColorStop(0.3, "#3ae7e9");
            grd.addColorStop(0.6, "#3dded0");
            grd.addColorStop(1, "#04decb");
            ctx.strokeStyle = grd;
        ctx.lineWidth=inW;
        ctx.arc(W/2,H/2,inR,0,Math.PI*2,false);
        ctx.stroke();

        //圆心背景
        ctx.beginPath();
        ctx.fillStyle="#b7fafe";
        ctx.arc(W/2,H/2,bgR,0,Math.PI*2,false);
        ctx.fill();

        //外层
        var r = deg*Math.PI/180;
        ctx.beginPath();
        var grd = ctx.createLinearGradient(0,H, W, 0);
            grd.addColorStop(0, "#f3453a");
            grd.addColorStop(0.3, "#f37e77");
            grd.addColorStop(0.6, "#f3685f");
            grd.addColorStop(1, "#f3857f");
            ctx.strokeStyle = grd;
        ctx.lineWidth=outW;
        ctx.arc(W/2,H/2,outR,0-90*Math.PI/180,r-90*Math.PI/180,false);
        ctx.stroke();
        //内层
        ctx.beginPath();
        var grd = ctx.createLinearGradient(W, 0, 0,H);
            grd.addColorStop(0, "#f3a5a0");
            grd.addColorStop(0.3, "#f3a5a0");
            grd.addColorStop(0.6, "#f39e99");
            grd.addColorStop(1, "#f39e99");
            ctx.strokeStyle = grd;
        ctx.lineWidth=inW;
        ctx.arc(W/2,H/2,inR,0-90*Math.PI/180,r-90*Math.PI/180,false);
        ctx.stroke();

        ctx.fillStyle="#000";
        ctx.font="bold "+R/4+"px Arial";
        text = Math.floor(deg/360*100)+"%";
        text_w = ctx.measureText(text).width;
        ctx.fillText(text,W/2 - text_w/2,H/2+R/10);
    };

    $this.ratePie=function(number){
          var i=0;
          var t= setInterval(function(){
            if(i==number){
                clearInterval(t);
            }else{
                number>0?i++:i--;
            }
            $this.ratePieNoAnimation(i);
            if(i > 100 || i < -100){//如果数字太大，取消动画效果
                $this.ratePieNoAnimation(number);
                clearInterval(t);
            }
          }, 1000/(number>0?number:-number));
    }
  }
