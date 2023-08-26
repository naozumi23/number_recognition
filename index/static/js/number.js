$(".openbtn").click(function () {//ボタンがクリックされたら
    $(this).toggleClass('active');//ボタン自身に activeクラスを付与し
    $("#g-nav").toggleClass('panelactive');//ナビゲーションにpanelactiveクラスを付与
});

$("#g-nav a").click(function () {//ナビゲーションのリンクがクリックされたら
    $(".openbtn").removeClass('active');//ボタンの activeクラスを除去し
    $("#g-nav").removeClass('panelactive');//ナビゲーションのpanelactiveクラスも除去
});

var canvas = document.getElementById('canvassample'),
    ctx = canvas.getContext('2d'),
    moveflg = 0,
    Xpoint,
    Ypoint,
    temp = [];

//初期値（サイズ、色、アルファ値）の決定
var defSize = 16,
    defColor = "#555";

// キャンバスを白色に塗る
ctx.fillStyle = 'rgb(255,255,255)';

// ストレージの初期化
var myStorage = localStorage;
window.onload = initLocalStorage();

// PC対応
canvas.addEventListener('mousedown', startPoint, false);
canvas.addEventListener('mousemove', movePoint, false);
canvas.addEventListener('mouseup', endPoint, false);
// スマホ対応
canvas.addEventListener('touchstart', startPoint, false);
canvas.addEventListener('touchmove', movePoint, false);
canvas.addEventListener('touchend', endPoint, false);

function startPoint(e) {
    e.preventDefault();
    ctx.beginPath();

    // 矢印の先っぽから始まるように調整
    Xpoint = e.layerX;
    Ypoint = e.layerY;

    ctx.moveTo(Xpoint, Ypoint);
}

function movePoint(e) {
    if (e.buttons === 1 || e.witch === 1 || e.type == 'touchmove') {
        Xpoint = e.layerX;
        Ypoint = e.layerY;
        moveflg = 1;

        ctx.lineTo(Xpoint, Ypoint);
        ctx.lineCap = "round";
        ctx.lineWidth = defSize * 2;
        ctx.strokeStyle = defColor;
        ctx.stroke();

    }
}

function endPoint(e) {
    if (moveflg === 0) {
        ctx.lineTo(Xpoint - 1, Ypoint - 1);
        ctx.lineCap = "round";
        ctx.lineWidth = defSize * 2;
        ctx.strokeStyle = defColor;
        ctx.stroke();

    }
    moveflg = 0;
    setLocalStoreage();
}

function clearCanvas() {
    if (confirm('Canvasを初期化しますか？')) {
        initLocalStorage();
        temp = [];
        resetCanvas();
    }
}

function resetCanvas() {
    ctx.clearRect(0, 0, ctx.canvas.clientWidth, ctx.canvas.clientHeight);
    ctx.fillStyle = 'rgb(255,255,255)';
}

function chgImg() {
    var png = canvas.toDataURL();
    document.getElementById("newImg").value = png;
}

function initLocalStorage() {
    myStorage.setItem("__log", JSON.stringify([]));
}

function setLocalStoreage() {
    var png = canvas.toDataURL();
    var logs = JSON.parse(myStorage.getItem("__log"));

    setTimeout(function () {
        logs.unshift({png: png});

        myStorage.setItem("__log", JSON.stringify(logs));

        temp = [];
    }, 0);
}


