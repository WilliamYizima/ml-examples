console.log('oi')

function response(intent){
    if (intent == "FazerTroca") return "você quer fazer uma troca!"
    if (intent == "None") return "Não entendi?!"
    if (intent == "MeusPedidos") return "VoC~e quer ver seus pedidos!"
}

function predict() {
    let input_message = document.querySelector('#minha-msg')
    let span_intent = document.querySelector('#intent-bot')
    let span_response = document.querySelector('#response-bot')

    let req_payload = { "sentence": input_message.value, "model_name": "modelo_nlp" }
    // let req_payload = { "number_day":5 }

    fetch('/analyze/nlp', {
        method: 'post',
        headers: {"Content-type": "application/json;charset=UTF-8"},
        body: JSON.stringify(req_payload)
    }).then(function (response) {
        return response.json();
    }).then(function (data) {
        console.log(data)
        span_intent.innerHTML = data.classifier
        span_response.innerHTML = response(data.classifier)   
    });
}