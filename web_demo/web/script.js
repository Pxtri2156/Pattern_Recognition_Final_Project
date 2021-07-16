

const audio = document.getElementById("audio")
const btn = document.getElementById("btn")
audio.addEventListener("change", (event) => {
    let file = audio.files[0]
    readAudioFile(file)
        .then(audioData => {
            let sound = document.createElement("audio");
            let source = document.createElement("source");
            sound.id = "audio-player";
            sound.controls = "controls";
            source.src = audioData;
            sound.type = "audio/mpeg";
            sound.appendChild(source);
            document.getElementById("preview").innerHTML = ""
            document.getElementById("preview").appendChild(sound);
        })

})
btn.addEventListener("click", clickButton)

renderProgress([20, 20, 20, 20, 20, 20])



function renderProgress(data) {
    const ids = [
        "angry-progress",
        "happy-progress",
        "exc-progress",
        "sad-progress",
        "fru-progress",
        "neu-progress"
    ]
    for (let i = 0; i < ids.length; i++) {
        let element = document.getElementById(ids[i])
        console.log(data[i])
        element.style.width = `${data[i]}%`
        element.innerHTML = `${data[i]}%`
    }

}

function readAudioFile(audioFile) {
    return new Promise((resolve, reject) => {
        let reader = new FileReader();
        reader.onload = function (evt) {
            let url = evt.target.result;
            resolve(url)
        };
        reader.onerror = ((event) => {
            reject(event.target.error)
        })
        reader.readAsDataURL(audioFile);
    })
}

function clickButton(event) {
    const config = {}
    const audio_model = document.getElementById("audio-model").value
    const image_model = document.getElementById("image-model").value
    try {

        if (audio_model == "null" && image_model == "null") {
            throw new Error("Select 1 model, audio model or visual model!")
        }
        if (audio_model != "null" && image_model != "null") {
            throw new Error("Select 1 model, audio model or visual model, not both!")
        }
        if (audio.files[0] ==  undefined){
            throw new Error("Please select an audio!")
        }
        if (audio_model != "null" && image_model == "null") {
            config["type"] = "audio"
            config["model"] = audio_model
        }
        if (audio_model == "null" && image_model != "null") {
            config["type"] = "image"
            config["model"] = image_model
        }
        predict(config,audio.files[0])
        .then(res=>res.json())
        .then(data=>{
            document.getElementById("spectrum-image").src = data.spectrum
            renderProgress(data.classifier)
            console.log(data)
        })
        
    } catch(error){
        alert(error.message)
        console.error(error);
    }
}

function predict(config,file){
    const body = new FormData()
    for (const key in config) {
        body.append(key,config[key])
      }
    body.append("audio",file)
    return fetch("/api/recognize",{
        method:"POST",
        body:body
    })
}
