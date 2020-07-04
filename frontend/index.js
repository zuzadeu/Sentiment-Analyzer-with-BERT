function textChanged(text) {
    console.log(text)
    fetch("http://127.0.0.1:5000/score", {
        method: "PUT", 
        body: JSON.stringify({"text": text}), 
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'},
    }).then(parseResponse)
}

function parseResponse(response){
    console.log(response)
    response.json().then(handleJSON)
}

function handleJSON(response){
    let score = (response.score * 100).toFixed(0)+'%'
    console.log(score)
    document.getElementById("score").textContent = score
    document.getElementById("circle").backgroundColor = colorsys.hsvToHex({h: response.score*120, s: 100, v: 100})
}