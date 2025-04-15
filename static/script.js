// Labels for the models
const labels = ['InceptionV3', 'DenseNet', 'VGG19'];

// Data for each metric
const accuracyData = [95, 94, 74];
const precisionData = [96, 94.5, 76];
const recallData = [95, 94, 74];
const f1ScoreData = [95.5, 94, 73];

// Creating the chart
const ctx = document.getElementById('metricsChart').getContext('2d');
new Chart(ctx, {
  type: 'bar',
  data: {
    labels: labels,
    datasets: [
      {
        label: 'Accuracy (%)',
        data: accuracyData,
        backgroundColor: 'rgba(54, 162, 235, 0.6)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1
      },
      {
        label: 'Precision (%)',
        data: precisionData,
        backgroundColor: 'rgba(75, 192, 192, 0.6)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1
      },
      {
        label: 'Recall (%)',
        data: recallData,
        backgroundColor: 'rgba(153, 102, 255, 0.6)',
        borderColor: 'rgba(153, 102, 255, 1)',
        borderWidth: 1
      },
      {
        label: 'F1-Score (%)',
        data: f1ScoreData,
        backgroundColor: 'rgba(255, 159, 64, 0.6)',
        borderColor: 'rgba(255, 159, 64, 1)',
        borderWidth: 1
      }
    ]
  },
  options: {
    responsive: true,
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          stepSize: 10
        }
      }
    },
    plugins: {
      title: {
        display: true,
        text: 'Model Performance Metrics Comparison'
      }
    }
  }
});
