addEventListener("DOMContentLoaded", main)

const TEMPLATES = {}

function main() {
    document.getElementById("user-input-form").addEventListener("submit", inputSubmitted);

    let templates = document.getElementById("chat-templates")
    TEMPLATES.userBox = templates.content.querySelector(".user-box").cloneNode(true);
    TEMPLATES.systemBox = templates.content.querySelector(".system-box").cloneNode(true);
}

async function inputSubmitted(e){
    e.preventDefault()

    let userInput = document.getElementById("user-input")

    let userBox = TEMPLATES.userBox.cloneNode(true)
    userBox.querySelector(".utterance").innerHTML = userInput.value
    document.getElementById("chat-window").appendChild(userBox)

    let chatWindow = document.getElementById("chat-window")
    chatWindow.scrollTop = chatWindow.scrollHeight

    userInput.value = "";

    let conversation = []
    let role = "user"
    for (let utterance of document.getElementsByClassName("utterance")) {
        conversation.push({"role": role, "content": utterance.textContent})

        if (role === "user"){
            role = "assistant"
        } else if (role === "assistant"){
            role = "user"
        }
    }

    console.log(conversation)
    await chat(conversation)
}

async function chat(conversation){
    let systemBox = TEMPLATES.systemBox.cloneNode(true)
    let utterance = systemBox.querySelector(".utterance")
    utterance.innerHTML = ""
    utterance.classList.add("loading")
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

        let decoded = textDecoder.decode(value)
        console.log(decoded)
        text += JSON.parse(decoded).text
        let utterance = systemBox.querySelector(".utterance")
        utterance.classList.remove("loading")
        utterance.innerHTML = converter.makeHtml(text)
        // utterance.scrollIntoView()
        let chatWindow = document.getElementById("chat-window")
        chatWindow.scrollTop = chatWindow.scrollHeight
    }
}