const samplePassengers = [
    {
        pclass: "1",
        name: "Carter, Miss. Helena",
        sex: "female",
        age: "28",
        sibsp: "0",
        parch: "0",
        fare: "80.00",
        cabin: "C85",
        embarked: "C",
    },
    {
        pclass: "3",
        name: "Nagy, Mr. Youssef",
        sex: "male",
        age: "23",
        sibsp: "1",
        parch: "0",
        fare: "12.50",
        cabin: "",
        embarked: "S",
    },
    {
        pclass: "2",
        name: "Miller, Mrs. Anna",
        sex: "female",
        age: "36",
        sibsp: "1",
        parch: "1",
        fare: "26.00",
        cabin: "",
        embarked: "S",
    },
];

function setValue(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.value = value ?? "";
    }
}

function fillForm(data) {
    setValue("pclass", data.pclass);
    setValue("name", data.name);
    setValue("sex", data.sex);
    setValue("age", data.age);
    setValue("sibsp", data.sibsp);
    setValue("parch", data.parch);
    setValue("fare", data.fare);
    setValue("cabin", data.cabin);
    setValue("embarked", data.embarked);

    window.scrollTo({ top: document.getElementById("predictionForm").offsetTop - 120, behavior: "smooth" });
}

const sampleBtn = document.getElementById("sampleBtn");
if (sampleBtn) {
    sampleBtn.addEventListener("click", () => {
        const randomPassenger = samplePassengers[Math.floor(Math.random() * samplePassengers.length)];
        fillForm(randomPassenger);
    });
}

document.querySelectorAll(".details-btn").forEach((button) => {
    button.addEventListener("click", () => {
        const targetId = button.dataset.target;
        const targetRow = document.getElementById(targetId);

        if (!targetRow) return;

        targetRow.classList.toggle("open");
        button.textContent = targetRow.classList.contains("open") ? "Hide" : "Details";
    });
});

document.querySelectorAll(".use-btn").forEach((button) => {
    button.addEventListener("click", () => {
        fillForm({
            pclass: button.dataset.pclass,
            name: button.dataset.name,
            sex: button.dataset.sex,
            age: button.dataset.age,
            sibsp: button.dataset.sibsp,
            parch: button.dataset.parch,
            fare: button.dataset.fare,
            cabin: button.dataset.cabin,
            embarked: button.dataset.embarked,
        });
    });
});
