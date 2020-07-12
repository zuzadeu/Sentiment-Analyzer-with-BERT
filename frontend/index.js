function textChangedHandler(text) {
    console.log(text)
    fetch("http://127.0.0.1:5000/score", {
        method: "PUT", 
        body: JSON.stringify({"text": text}), 
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'},
    }).then(parseResponse)
}

const textChanged = debounce(textChangedHandler, 250);

function parseResponse(response){
    console.log(response)
    response.json().then(handleJSON)
}

function handleJSON(response){
    let score = (response.score * 100).toFixed(0)+'%'
    console.log(score)
    document.getElementById("score").textContent = score
    document.getElementById("circle").style.backgroundColor = hsv2hex({h: response.score*120, s: 100, v: 80}) 
}

function debounce(func, wait, immediate) {
	var timeout;
	return function() {
		var context = this, args = arguments;
		var later = function() {
			timeout = null;
			if (!immediate) func.apply(context, args);
		};
		var callNow = immediate && !timeout;
		clearTimeout(timeout);
		timeout = setTimeout(later, wait);
		if (callNow) func.apply(context, args);
	};
};

function hsv2hex(h, s, v) {
    var rgb = hsv2Rgb(h, s, v)
    return rgb2Hex(rgb.r, rgb.g, rgb.b)
  }

function hsv2Rgb(h, s, v) {
    if (typeof h === 'object') {
      const args = h
      h = args.h; s = args.s; v = args.v;
    }
  
    h = _normalizeAngle(h)
    h = (h === HUE_MAX) ? 1 : (h % HUE_MAX / parseFloat(HUE_MAX) * 6)
    s = (s === SV_MAX) ? 1 : (s % SV_MAX / parseFloat(SV_MAX))
    v = (v === SV_MAX) ? 1 : (v % SV_MAX / parseFloat(SV_MAX))
  
    var i = Math.floor(h)
    var f = h - i
    var p = v * (1 - s)
    var q = v * (1 - f * s)
    var t = v * (1 - (1 - f) * s)
    var mod = i % 6
    var r = [v, q, p, p, t, v][mod]
    var g = [t, v, v, q, p, p][mod]
    var b = [p, p, t, v, v, q][mod]
  
    return {
      r: Math.floor(r * RGB_MAX),
      g: Math.floor(g * RGB_MAX),
      b: Math.floor(b * RGB_MAX),
    }
  }

function rgb2Hex(r, g, b) {
    if (typeof r === 'object') {
      const args = r
      r = args.r; g = args.g; b = args.b;
    }
    r = Math.round(r).toString(16)
    g = Math.round(g).toString(16)
    b = Math.round(b).toString(16)
  
    r = r.length === 1 ? '0' + r : r
    g = g.length === 1 ? '0' + g : g
    b = b.length === 1 ? '0' + b : b
  
    return '#' + r + g + b
  }

function _normalizeAngle (degrees) {
    return (degrees % 360 + 360) % 360;
  }

const RGB_MAX = 255
const HUE_MAX = 360
const SV_MAX = 100