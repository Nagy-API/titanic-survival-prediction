"use strict";

const form = document.getElementById("predictionForm");
const fields = document.getElementById("passenger-fields");
const predictButton = document.getElementById("predictBtn");
const sampleButton = document.getElementById("sampleBtn");
const statusMessage = document.getElementById("prediction-status");
const error = document.getElementById("prediction-error");
const result = document.getElementById("prediction-result");
const usedInputs = document.getElementById("used-inputs");
const heading = document.getElementById("prediction-label");
const percentage = document.getElementById("prediction-percentage");
const meter = document.getElementById("probability-meter");

const samples = [
    { pclass: "1", sex: "female", title: "Miss", age: "28", fare: "80", sibsp: "0", parch: "0", cabin: "true", embarked: "C" },
    { pclass: "3", sex: "male", title: "Mr", age: "23", fare: "12.50", sibsp: "1", parch: "0", cabin: "false", embarked: "S" },
    { pclass: "2", sex: "female", title: "Mrs", age: "36", fare: "26", sibsp: "1", parch: "1", cabin: "false", embarked: "S" },
];
let sampleIndex = 0;
let pending = false;

function setStatus(message) {
    statusMessage.hidden = !message;
    if (statusMessage.textContent !== message) statusMessage.textContent = message;
}

function clearResult(message = "Submit a passenger profile to see the model estimate.") {
    result.hidden = true;
    usedInputs.replaceChildren();
    error.textContent = "";
    setStatus(message);
}

function selectLabel(name) {
    const field = form.elements.namedItem(name);
    return field.selectedOptions[0].textContent;
}

function readProfile() {
    const values = new FormData(form);
    const age = values.get("age");
    const fare = values.get("fare");
    return {
        Pclass: Number(values.get("pclass")),
        Sex: values.get("sex"),
        Title: values.get("title"),
        Age: age === "" ? null : Number(age),
        Fare: fare === "" ? null : Number(fare),
        SibSp: Number(values.get("sibsp")),
        Parch: Number(values.get("parch")),
        HasCabin: values.get("cabin") === "true",
        Embarked: values.get("embarked") || null,
    };
}

function summarizeProfile(profile) {
    return [
        ["Class", selectLabel("pclass")],
        ["Sex", selectLabel("sex")],
        ["Title", selectLabel("title")],
        ["Age", profile.Age === null ? "Unknown" : String(profile.Age)],
        ["Fare", profile.Fare === null ? "Unknown" : `£${profile.Fare.toFixed(2)}`],
        ["Family size", String(profile.SibSp + profile.Parch + 1)],
        ["Cabin recorded", selectLabel("cabin")],
        ["Embarkation", selectLabel("embarked")],
    ];
}

function renderResult(data, summary) {
    const score = data.survival_percentage;
    if (![0, 1].includes(data.prediction_code) || !Number.isFinite(score) || score < 0 || score > 100) {
        throw new Error("Unexpected prediction response");
    }

    const survived = data.prediction_code === 1;
    heading.textContent = survived ? "Survived" : "Did not survive";
    percentage.textContent = `${score.toFixed(1)}%`;
    meter.value = score;
    result.classList.toggle("result-survived", survived);
    result.classList.toggle("result-danger", !survived);
    result.querySelector("details").open = false;

    for (const [label, value] of summary) {
        const row = document.createElement("div");
        const term = document.createElement("dt");
        const description = document.createElement("dd");
        term.textContent = label;
        description.textContent = value;
        row.append(term, description);
        usedInputs.append(row);
    }
    setStatus("");
    result.hidden = false;
    result.focus();
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (pending) return;

    const profile = readProfile();
    const summary = summarizeProfile(profile);
    pending = true;
    fields.disabled = true;
    predictButton.textContent = "Estimating…";
    clearResult("Estimating survival for this profile…");

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(profile),
            signal: AbortSignal.timeout(15000),
        });
        if (!response.ok) throw new Error("Prediction unavailable");
        renderResult(await response.json(), summary);
    } catch {
        setStatus("");
        error.textContent = "The estimate could not be completed. Check your entries and try again.";
    } finally {
        pending = false;
        fields.disabled = false;
        predictButton.textContent = "Predict survival";
        if (error.textContent && document.activeElement === document.body) predictButton.focus();
    }
});

sampleButton.addEventListener("click", () => {
    for (const [name, value] of Object.entries(samples[sampleIndex])) {
        form.elements.namedItem(name).value = value;
    }
    sampleIndex = (sampleIndex + 1) % samples.length;
    clearResult("Sample profile loaded. Select Predict survival to continue.");
});

form.addEventListener("reset", () => {
    sampleIndex = 0;
    clearResult();
});
form.addEventListener("input", () => clearResult());

fields.disabled = false;
