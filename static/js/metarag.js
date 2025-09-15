addEventListener("DOMContentLoaded", main)

const TEMPLATES = {}

function main() {
    document.getElementById("user-input-form").addEventListener("submit", inputSubmitted);

    let templates = document.getElementById("chat-templates")
    TEMPLATES.userBox = templates.content.querySelector(".user-box").cloneNode(true);
    TEMPLATES.systemBox = templates.content.querySelector(".system-box").cloneNode(true);
    console.log(TEMPLATES);
}

async function inputSubmitted(e){
    e.preventDefault()

    let userInput = document.getElementById("user-input")

    let userBox = TEMPLATES.userBox.cloneNode(true)
    userBox.querySelector(".utterance").innerHTML = userInput.value
    document.getElementById("chat-window").appendChild(userBox)

    userInput.value = "";

    let conversation = []
    let role = "user"
    for (let utterance of document.getElementsByClassName("utterance")) {
        conversation.push({"role": role, "content": utterance.textContent})

        if (role === "user"){
            role = "system"
        }
    }

    await chat(conversation)
}

async function chat(conversation){
    let systemBox = TEMPLATES.systemBox.cloneNode(true)
    systemBox.querySelector(".utterance").innerHTML = ""
    document.getElementById("chat-window").appendChild(systemBox)

    const response = await fetch("chat",
        {
            method: "POST",
            body: JSON.stringify(conversation),
            headers: {
                "Content-Type": "application/json"
            }
        }
    )
    const converter = new showdown.Converter()
    const textDecoder = new TextDecoder()
    const reader = response.body.getReader()
    let text = ""
    while (true) {
        ({value, done} = await reader.read())
        if(done){
            return
        }

        text += JSON.parse(textDecoder.decode(value)).text
        systemBox.querySelector(".utterance").innerHTML = converter.makeHtml(text)
    }
}