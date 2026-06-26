let chart, model, map;

// -----------------------------
// Signal Generator
// -----------------------------
function generateSignal(type) {
  let data = [];
  for (let i = 0; i < 200; i++) {
    if (type === "footstep") {
      data.push(Math.sin(i/8) * 0.7 + Math.random()*0.1);
    } else if (type === "vehicle") {
      data.push(Math.sin(i/2) + Math.random()*0.3);
    } else if (type === "heavyVehicle") {
      data.push(Math.sin(i/1.5) * 2 + Math.random()*0.5);
    } else if (type === "animal") {
      data.push(Math.sin(i/4) + Math.random()*0.25);
    } else if (type === "insect") {
      data.push(Math.sin(i/12) * 0.2 + Math.random()*0.05);
    }
  }
  return data;
}

// -----------------------------
// Feature Extraction
// -----------------------------
function extractFeatures(signal) {
  let mean = signal.reduce((a,b)=>a+b,0)/signal.length;
  let variance = signal.map(x => (x-mean)**2).reduce((a,b)=>a+b,0)/signal.length;
  let max = Math.max(...signal);
  let peaks = signal.filter((v,i,arr) => i>0 && i<arr.length-1 && v>arr[i-1] && v>arr[i+1]).length;
  let rhythm = signal.slice(1).map((v,i)=>Math.abs(v-signal[i])).reduce((a,b)=>a+b,0)/signal.length;
  return [mean, variance, max, peaks, rhythm];
}

// -----------------------------
// Train AI Model
// -----------------------------
async function trainModel() {
  model = tf.sequential();
  model.add(tf.layers.dense({units: 12, activation: 'relu', inputShape: [5]}));
  model.add(tf.layers.dense({units: 5, activation: 'softmax'}));
  model.compile({optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy']});

  let X = [], y = [];
  let labels = {
    "footstep": [1,0,0,0,0],
    "vehicle": [0,1,0,0,0],
    "heavyVehicle": [0,0,1,0,0],
    "animal": [0,0,0,1,0],
    "insect": [0,0,0,0,1]
  };

  for (let label in labels) {
    for (let i=0; i<400; i++) {
      let sig = generateSignal(label);
      let features = extractFeatures(sig);
      X.push(features);
      y.push(labels[label]);
    }
  }

  const xs = tf.tensor2d(X);
  const ys = tf.tensor2d(y);
  await model.fit(xs, ys, {epochs: 20});
}
trainModel();

// -----------------------------
// Detect Activity (GLOBAL)
// -----------------------------
window.detectActivity = async function detectActivity() {
  if (!model) {
    document.getElementById("result").innerText = "⚠️ Model not ready yet!";
    return;
  }

  const types = ["footstep", "vehicle", "heavyVehicle", "animal", "insect"];
  const labels = ["Human Footstep", "Vehicle", "Heavy Vehicle", "Animal", "Insect"];

  const typeIndex = Math.floor(Math.random() * types.length);
  const type = types[typeIndex];

  let signal = generateSignal(type);
  let features = extractFeatures(signal);

  const predictionTensor = model.predict(tf.tensor2d([features]));
  const predictionArray = await predictionTensor.data();
  const predictionIndex = predictionArray.indexOf(Math.max(...predictionArray));

  document.getElementById("graphContainer").style.display = "block";

  const colorMap = {
    "footstep": "#1f77b4",
    "vehicle": "#ff7f0e",
    "heavyVehicle": "#d62728",
    "animal": "#2ca02c",
    "insect": "#9467bd"
  };
  const lineColor = colorMap[type];

  if (!chart) {
    chart = new Chart(document.getElementById("signalChart"), {
      type: 'line',
      data: {
        labels: Array.from({length: signal.length}, (_, i) => i),
        datasets: [{
          label: 'Signal',
          data: signal,
          borderColor: lineColor,
          borderWidth: 2,
          fill: false
        }]
      },
      options: { responsive: true }
    });
  } else {
    chart.data.datasets[0].data = signal;
    chart.data.datasets[0].borderColor = lineColor;
    chart.update();
  }

  document.getElementById("result").innerText = "Suspected Activity: " + labels[predictionIndex];

  if (features[2] > 2.0) {
    triggerAlert(`Heavy activity detected (${labels[predictionIndex]})`, "high");
  } else if (features[2] > 1.0) {
    triggerAlert(`Moderate activity detected (${labels[predictionIndex]})`, "medium");
  } else {
    triggerAlert("", "low");
  }

  updateLocationMap(labels[predictionIndex]);
};

// -----------------------------
// Alert + Beep
// -----------------------------
function triggerAlert(message, level) {
  const beep = document.getElementById("beepSound");
  const alertBox = document.getElementById("alertBox");
  const resultElement = document.getElementById("result");

  if (level === "high") {
    resultElement.style.color = "red";
    resultElement.style.fontWeight = "bold";
    resultElement.innerText = message;
    alertBox.innerText = message;
    beep.play();
  } else if (level === "medium") {
    resultElement.style.color = "orange";
    resultElement.innerText = message;
    alertBox.innerText = message;
    beep.play();
  } else {
    resultElement.style.color = "#ee9b00";
    alertBox.innerText = "Normal activity";
  }
}

// -----------------------------
// Map Initialization + Location
// -----------------------------
document.addEventListener("DOMContentLoaded", () => {
  map = L.map('map').setView([28.45, 78.45], 15); // default coords
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
  }).addTo(map);

  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const lat = pos.coords.latitude;
        const lng = pos.coords.longitude;
        map.setView([lat, lng], 16);
        L.marker([lat, lng]).addTo(map).bindPopup("📍 You are here").openPopup();
      },
      () => {
        console.log("Location access denied, using default coords.");
      }
    );
  }
});

// -----------------------------
// Update Map with Detection
// -----------------------------
function updateLocationMap(activityType) {
  const center = map.getCenter();
  const lat = center.lat + (Math.random() - 0.5) * 0.002;
  const lng = center.lng + (Math.random() - 0.5) * 0.002;

  const colorMap = {
    "Human Footstep": "blue",
    "Vehicle": "orange",
    "Heavy Vehicle": "red",
    "Animal": "green",
    "Insect": "purple"
  };

  let marker = L.circleMarker([lat, lng], {
    radius: 10,
    color: colorMap[activityType] || "black",
    fillColor: colorMap[activityType] || "black",
    fillOpacity: 0.8
  }).addTo(map);

  marker.bindPopup(`Detected: ${activityType}`).openPopup();
}


// -----------------------------
// Compare Footsteps
// -----------------------------
// -----------------------------
// Compare Any Two Activities
// -----------------------------
window.compareActivities = function compareActivities() {
  const activity1 = document.getElementById("activity1").value;
  const activity2 = document.getElementById("activity2").value;

  // Generate signals
  let signal1 = generateSignal(activity1);
  let signal2 = generateSignal(activity2);

  // Extract features
  let features1 = extractFeatures(signal1);
  let features2 = extractFeatures(signal2);

  // Show chart
  document.getElementById("comparisonContainer").style.display = "block";

  if (!window.comparisonChart) {
    window.comparisonChart = new Chart(document.getElementById("comparisonChart"), {
      type: 'line',
      data: {
        labels: Array.from({length: signal1.length}, (_, i) => i),
        datasets: [
          { label: activity1, data: signal1, borderColor: 'blue', fill: false },
          { label: activity2, data: signal2, borderColor: 'green', fill: false }
        ]
      },
      options: { responsive: true }
    });
  } else {
    window.comparisonChart.data.datasets[0].label = activity1;
    window.comparisonChart.data.datasets[0].data = signal1;
    window.comparisonChart.data.datasets[1].label = activity2;
    window.comparisonChart.data.datasets[1].data = signal2;
    window.comparisonChart.update();
  }

  // Compare features
  let comparisonText = `
    ${activity1} → Mean: ${features1[0].toFixed(2)}, Variance: ${features1[1].toFixed(2)}, Peaks: ${features1[3]}, Rhythm: ${features1[4].toFixed(2)}
    ${activity2} → Mean: ${features2[0].toFixed(2)}, Variance: ${features2[1].toFixed(2)}, Peaks: ${features2[3]}, Rhythm: ${features2[4].toFixed(2)}
  `;

  // Interpretation
  let interpretation = "";
  if (features1[2] > features2[2]) {
    interpretation = `${activity1} shows stronger signal amplitude than ${activity2}.`;
  } else if (features2[2] > features1[2]) {
    interpretation = `${activity2} shows stronger signal amplitude than ${activity1}.`;
  } else {
    interpretation = `Both activities have similar amplitude levels.`;
  }

  document.getElementById("comparisonResult").innerText = comparisonText + "\n\n" + interpretation;
};

