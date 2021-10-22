var cvs0 = null;
var cvs1 = null;
var cvs2 = null;
var portrait_dataurl = '';
var portrait_w = 0;
var portrait_h = 0;
var bboxes = null;
var cur_bbox_id = 0;
var is_spouse = false;
var got_me = false;
var got_spouse = false;
var b64_face_me = '';
var b64_face_spouse = '';

function detect(i) {
    let input_id = 'face';
    input_id += i;
    const input = document.getElementById(input_id);
    if (!input) {
        alert(`Cannot find input${i}!`);
        return;
    }

    if (i === 0) {
        is_spouse = false;
    } else if (i === 1) {
        is_spouse = true;
    } else {
        alert(`Error number ${i}!`);
        return;
    }

    const file = input.files[0];
    if (!file) {
        alert('No file selected!');
        return;
    }

    const reader = new FileReader();
    reader.addEventListener('load', function () {
        portrait_dataurl = reader.result;

        const ctx = cvs0.getContext('2d');
        const img = new Image();
        img.onload = function (e) {
            cvs0.width = portrait_w = img.naturalWidth;
            cvs0.height = portrait_h = img.naturalHeight;
            ctx.drawImage(img, 0, 0);
        }
        img.src = portrait_dataurl;

        fetch(DETECT_URL, {
            method: 'POST',
            credentials: 'omit',
            cache: 'no-cache',
            headers: new Headers({
                'content-type': 'application/json'
            }),
            body: JSON.stringify({'portrait':portrait_dataurl.split(',')[1]})
        }).then(function (response) {
            if (response.status !== 200) {
                throw 'DETECT ERROR: ' + response.status;
            }
            return response.json();
        }).then(function (data) {
            const error_msg = data['error'];
            if (error_msg) {
                throw 'DETECT ERROR: ' + error_msg;
            }

            bboxes = data['bboxes'];
            if (bboxes.length === 0) {
                throw 'DETECT ERROR: Cannot detect any human face!';
            } else if (bboxes.length === 1) {
                align(bboxes[0]);
            } else {
                select_face(i, bboxes);
            }
        }).catch(function (error) {
            alert(error);
        })
    })
    reader.readAsDataURL(file);
}

function select_face(i, bboxes) {
    const div_couple = document.getElementsByClassName('page1')[0];
    div_couple.style.display = 'none';
    const div_result = document.getElementsByClassName('page-result')[0];
    div_result.style.display = 'none';
    const div_select = document.getElementsByClassName('page2')[0];
    div_select.style.display = 'block';
    const tohide = document.getElementsByClassName('tohide')[0];
    if (i) {
        tohide.style.display = 'inline';
    } else {
        tohide.style.display = 'none';
    }
    
    cur_bbox_id = 0;
    _redraw(cur_bbox_id, bboxes);
}

function _redraw(cur_bbox_id, bboxes) {
    const ctx = cvs0.getContext('2d');
    const img = new Image();
    img.onload = function (e) {
        ctx.drawImage(img, 0, 0);

        ctx.save();
        ctx.globalCompositeOperation = 'hard-light';
        ctx.fillStyle = '#555555';
        ctx.fillRect(0, 0, cvs0.width, cvs0.height);
        for (let i in bboxes) {
            const bbox = bboxes[i];
            const [x0,y0,x1,y1] = bbox;
            const [w,h] = [x1-x0+1, y1-y0+1];
            if (i == cur_bbox_id) {
                ctx.strokeStyle = '#ffff00';
                ctx.fillStyle = '#444444';
            } else {
                ctx.strokeStyle = '#00ff00';
                ctx.fillStyle = '#222222';
            }
            ctx.lineWidth = 2;
            ctx.globalCompositeOperation = 'source-over';
            ctx.strokeRect(x0-1, y0-1, w+2, h+2);
            ctx.globalCompositeOperation = 'lighter';
            ctx.fillRect(x0, y0, w, h);
        }
        ctx.restore();
    }
    img.src = portrait_dataurl;
}

function callback_select(ev) {
    let [x, y] = [0, 0];
    if (ev.offsetX == null) {
        [x, y] = [ev.layerX, ev.layerY];
    } else {
        [x, y] = [ev.offsetX, ev.offsetY];
    }

    for (let i in bboxes) {
        const bbox = bboxes[i];
        const [x0,y0,x1,y1] = bbox;
        if (x0 <=x && x<=x1 && y0<=y &&y<=y1) {
            cur_bbox_id = i;
            _redraw(cur_bbox_id, bboxes);
            break;
        }
    }
}

function _end_select() {
    const div_couple = document.getElementsByClassName('page1')[0];
    div_couple.style.display = 'block';
    const div_result = document.getElementsByClassName('page-result')[0];
    div_result.style.display = 'block';
    const div_select = document.getElementsByClassName('page2')[0];
    div_select.style.display = 'none';
}

function cancel() {
    _end_select();
}

function select() {
    _end_select();

    align(bboxes[cur_bbox_id]);
}

function align(bbox) {
    const [x0,y0,x1,y1] = bbox;
    const h = y1 - y0 + 1;
    const w = x1 - x0 + 1;
    var max_side = Math.max(h, w);
    max_side += Math.floor(max_side / 2);

    const px0 = x0 + Math.floor(w / 2) - Math.floor(max_side / 2);
    const py0 = y0 + Math.floor(h / 2) - Math.floor(max_side / 2);
    const px1 = px0 + max_side - 1;
    const py1 = py0 + max_side - 1;

    const sx0 = Math.max(px0, 0);
    const sx1 = Math.min(px1, portrait_w - 1);
    const sw = sx1 - sx0 + 1;
    const dx0 = Math.max(-px0, 0);
    const dw = sw;
    const sy0 = Math.max(py0, 0);
    const sy1 = Math.min(py1, portrait_h - 1);
    const sh = sy1 - sy0 + 1;
    const dy0 = Math.max(-py0, 0);
    const dh = sh;

    cvs3.width = max_side;
    cvs3.height = max_side;
    const ctx = cvs3.getContext('2d');
    const img = new Image();
    img.onload = function (e) {
        ctx.clearRect(0, 0, cvs3.width, cvs3.height);
        ctx.drawImage(img, sx0, sy0, sw, sh, dx0, dy0, dw, dh);
        const face_dataurl = cvs3.toDataURL('image/png');
        const bx0 = x0 - px0;
        const by0 = y0 - py0;
        const bx1 = x1 - px0;
        const by1 = y1 - py0;
        const box = [bx0,by0,bx1,by1];

        fetch(ALIGN_URL, {
            method: 'POST',
            credentials: 'omit',
            cache: 'no-cache',
            headers: new Headers({
                'content-type': 'application/json'
            }),
            body: JSON.stringify({'patch':face_dataurl.split(',')[1], 'bbox': box})
        }).then(function (response) {
            if (response.status !== 200) {
                throw 'ALIGN ERROR: ' + response.status;
            }
            return response.json();
        }).then(function (data) {
            const error_msg = data['error'];
            if (error_msg) {
                throw 'ALIGN ERROR: ' + error_msg;
            }

            const b64_face_raw = data['face'];
            if ( (!b64_face_raw) && (data['aligned'] === false) ) {
                throw 'ALIGN ERROR: Cannot align the face!';
            }

            var cvs = null;
            if (is_spouse) {
                cvs = cvs2;
                got_spouse = true;
                b64_face_spouse = b64_face_raw;
            } else {
                cvs = cvs1;
                got_me = true;
                b64_face_me = b64_face_raw;
            }

            const ctx = cvs.getContext('2d');
            const img = new Image();
            img.onload = function (e) {
                ctx.clearRect(0, 0, cvs.width, cvs.height);
                ctx.drawImage(img, 0, 0);
            }
            img.src = "data:image/png;base64," + b64_face_raw;

            clear_result();
        }).catch(function (error) {
            alert(error);
        })
    }
    img.src = portrait_dataurl;
}

function clear_result() {
    const div_result = document.getElementsByClassName('page-result')[0];
    div_result.style.display = 'none';
}

function simi() {
    if (!got_me) {
        alert('Please choose a photo containing your face first.');
        return;
    } else if (!got_spouse) {
        alert("Please choose a photo containing your spouse's face first.");
        return;
    }

    cal_result();
}

function cal_result() {
    const btn = document.getElementById('btn-simi');
    btn.disabled = true;
    btn.innerHTML = 'Calculating...';

    fetch(SIMI_URL, {
        method: 'POST',
        credentials: 'omit',
        cache: 'no-cache',
        headers: new Headers({
            'content-type': 'application/json'
        }),
        body: JSON.stringify({'face_me': b64_face_me, 'face_spouse': b64_face_spouse, 'aligned': true})
    }).then(function (response) {
        if (response.status !== 200) {
            throw 'CALCULATE ERROR: ' + response.status;
        }
        return response.json();
    }).then(function (data) {
        const error_msg = data['error'];
        if (error_msg) {
            throw 'CALCULATE ERROR: ' + error_msg;
        }

        const formatter = Intl.NumberFormat('en-US', {
            style: 'percent',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
          })

        // document.getElementById('eye_simi').innerHTML = formatter.format(data['eye_simi']);
        // document.getElementById('nose_simi').innerHTML = formatter.format(data['nose_simi']);
        // document.getElementById('mouth_simi').innerHTML = formatter.format(data['mouth_simi']);
        document.getElementById('face_simi').innerHTML = formatter.format(data['face_simi']);
        // document.getElementById('syn_simi').innerHTML = formatter.format(data['syn_simi']);

        const div_result = document.getElementsByClassName('page-result')[0];
        div_result.style.display = 'block';

        btn.disabled = false;
        btn.innerHTML = 'Calculate the Similarities of Couple&#39;s Faces';
    }).catch(function (error) {
        alert(error);
    })
}

function load() {
    cvs1 = document.getElementById("cvs1");
    cvs2 = document.getElementById("cvs2");
    for (let cvs of [cvs1, cvs2]) {
        cvs.width = 112;
        cvs.height = 112;
        const ctx = cvs.getContext('2d');
        const img = new Image();
        img.onload = function (e) {
            ctx.drawImage(img, 0, 0);
        }
        img.src = DEFAULT_PHOTO;
    }

    cvs0 = document.getElementById("cvs0");
    cvs0.addEventListener('click', callback_select);
}

document.addEventListener("DOMContentLoaded", load);
