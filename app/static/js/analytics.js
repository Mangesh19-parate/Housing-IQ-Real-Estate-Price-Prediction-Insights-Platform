/**
 * HousingIQ - Analytics Dashboard JS
 * Fetches precomputed cache from /api/analytics/<chart_id>?city=<city>
 * Renders with Chart.js (charts) and Leaflet (spatial map).
 */

(function () {
  'use strict';

  // Chart type -> renderer function mapping
  const renderers = {
    map: renderSpatialMap,
    box: renderBoxPlot,
    scatter: renderScatter,
    line: renderLineChart,
    bar: renderBarChart,
    pie: renderPieChart,
    heatmap: renderHeatmap,
    wordcloud: renderWordCloud,
  };

  // Cache Chart.js instances for proper cleanup on city change
  const chartInstances = new Map();
  let spatialMap = null;

  // DOM elements
  const cityFilter = document.getElementById('city-filter');
  const filterStatus = document.getElementById('city-filter-status');
  const chartTiles = document.querySelectorAll('.chart-tile[data-chart-id]');
  const dashboard = document.querySelector('.analytics-dashboard');

  // API base from data attribute on dashboard section
  const API_BASE = dashboard?.dataset?.apiBase || '';

  // Current city state
  let currentCity = 'ALL';

  // Chart configurations (consistent colors from CSS vars)
  const CHART_COLORS = {
    primary: 'hsl(210, 90%, 50%)',
    primaryLight: 'hsla(210, 90%, 50%, 0.15)',
    secondary: 'hsl(140, 60%, 40%)',
    muted: 'hsl(220, 15%, 45%)',
    border: 'hsl(220, 15%, 88%)',
    grid: 'hsl(220, 15%, 92%)',
    fg: 'hsl(220, 25%, 15%)',
    cardBg: 'hsl(0, 0%, 100%)',
  };

  // Initialize on DOM ready
  document.addEventListener('DOMContentLoaded', init);

  async function init() {
    if (!cityFilter || chartTiles.length === 0) return;

    // Load all charts for initial city
    await loadAllCharts(currentCity);

    // City filter change handler
    cityFilter.addEventListener('change', async (e) => {
      currentCity = e.target.value;
      filterStatus.textContent = 'Loading ' + (currentCity === 'ALL' ? 'all cities' : currentCity) + '...';
      await loadAllCharts(currentCity);
      filterStatus.textContent = 'Showing ' + (currentCity === 'ALL' ? 'all cities' : currentCity);
    });
  }

  async function loadAllCharts(city) {
    const promises = Array.from(chartTiles).map((tile) => {
      const chartId = tile.dataset.chartId;
      return loadChart(chartId, city);
    });

    await Promise.allSettled(promises);
  }

  async function loadChart(chartId, city) {
    const tile = document.querySelector('.chart-tile[data-chart-id="' + chartId + '"]');
    if (!tile) return;

    const loadingEl = tile.querySelector('.chart-tile__loading');
    const errorEl = tile.querySelector('.chart-tile__error');
    const canvas = tile.querySelector('canvas');
    const mapContainer = tile.querySelector('.chart-map');
    const wordCloudContainer = tile.querySelector('.chart-wordcloud');

    // Show loading state
    if (loadingEl) loadingEl.hidden = false;
    if (errorEl) errorEl.hidden = true;
    if (canvas) canvas.style.opacity = '0';
    if (mapContainer) mapContainer.style.opacity = '0';
    if (wordCloudContainer) wordCloudContainer.style.opacity = '0';

    try {
      const response = await fetch(API_BASE + '/api/analytics/' + chartId + '?city=' + encodeURIComponent(city));
      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(err.detail || 'HTTP ' + response.status);
      }
      const data = await response.json();

      // Extract the city slice data
      const slice = data.city_slices?.[city];
      if (!slice) {
        throw new Error('No data for city "' + city + '"');
      }

      // Dispatch to appropriate renderer
      const renderer = renderers[data.chart_type];
      if (!renderer) {
        throw new Error('Unknown chart type: ' + data.chart_type);
      }

      await renderer(tile, slice, data.chart_type, data.title);

      // Hide loading, show chart
      if (loadingEl) loadingEl.hidden = true;
      if (canvas) canvas.style.opacity = '1';
      if (mapContainer) mapContainer.style.opacity = '1';
      if (wordCloudContainer) wordCloudContainer.style.opacity = '1';

    } catch (err) {
      console.error('Failed to load ' + chartId + ':', err);
      if (loadingEl) loadingEl.hidden = true;
      if (errorEl) {
        errorEl.hidden = false;
        errorEl.textContent = 'Failed to load: ' + err.message;
      }
    }
  }

  // ==================== RENDERERS ====================

  function renderSpatialMap(tile, slice, chartType, title) {
    const mapEl = tile.querySelector('.chart-map');
    if (!mapEl || !slice.markers?.length) {
      mapEl.textContent = 'No map data available';
      return;
    }

    // Destroy existing map if any
    if (spatialMap) {
      spatialMap.remove();
      spatialMap = null;
    }

    // Initialize Leaflet map
    const markers = slice.markers;
    const bounds = slice.bounds || calculateBounds(markers);

    spatialMap = L.map(mapEl).fitBounds(bounds);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors',
    }).addTo(spatialMap);

    // Color scale for price per sqft
    const prices = markers.map(function(m) { return m.price_per_sqft; }).filter(function(p) { return p > 0; });
    const minPrice = Math.min.apply(null, prices);
    const maxPrice = Math.max.apply(null, prices);

    function getColor(price) {
      if (!price || price <= 0) return '#999';
      var ratio = (price - minPrice) / (maxPrice - minPrice || 1);
      var hue = 120 - ratio * 120; // Green (low) to Red (high)
      return 'hsl(' + hue + ', 70%, 45%)';
    }

    markers.forEach(function(marker) {
      if (!marker.lat || !marker.lng) return;
      var circle = L.circleMarker([marker.lat, marker.lng], {
        radius: 6,
        fillColor: getColor(marker.price_per_sqft),
        color: '#fff',
        weight: 1,
        opacity: 1,
        fillOpacity: 0.8,
      });

      var priceLakh = (marker.price_inr / 100000).toFixed(1);
      var ppsf = marker.price_per_sqft ? Math.round(marker.price_per_sqft).toLocaleString() : 'N/A';
      circle.bindPopup(
        '<strong>' + (marker.locality || 'Unknown') + '</strong><br>' +
        'Rs' + priceLakh + ' L - ' + (marker.bhk || '?') + ' BHK<br>' +
        'Rs' + ppsf + '/sqft'
      );

      circle.addTo(spatialMap);
    });

    // Handle map resize on tile visibility
    setTimeout(function() { spatialMap.invalidateSize(); }, 100);
  }

  function renderBoxPlot(tile, slice, chartType, title) {
    var canvas = tile.querySelector('canvas');
    if (!canvas) return;

    var ctx = canvas.getContext('2d');
    destroyChart(chartIdFromTile(tile));

    var labels = slice.labels || [];
    var series = slice.series || []; // Array of arrays: [min, q1, median, q3, max, outliers...]
    var counts = slice.counts || [];

    var datasets = series.map(function(data, i) {
      return {
        label: labels[i] || 'Series ' + (i + 1),
        data: data,
        backgroundColor: CHART_COLORS.primaryLight,
        borderColor: CHART_COLORS.primary,
        borderWidth: 1,
        outlierColor: CHART_COLORS.muted,
        outlierRadius: 3,
      };
    });

    var chart = new Chart(ctx, {
      type: 'boxplot',
      data: { labels: labels, datasets: datasets },
      options: boxPlotOptions(title, counts),
    });

    chartInstances.set(chartIdFromTile(tile), chart);
  }

  function renderScatter(tile, slice, chartType, title) {
    var canvas = tile.querySelector('canvas');
    if (!canvas) return;

    var ctx = canvas.getContext('2d');
    destroyChart(chartIdFromTile(tile));

    var x = slice.x || [];
    var y = slice.y || [];
    var trend = slice.trend;

    var datasets = [{
      label: 'Listings',
      data: x.map(function(xi, i) { return { x: xi, y: y[i] }; }),
      backgroundColor: CHART_COLORS.primaryLight,
      borderColor: CHART_COLORS.primary,
      pointRadius: 4,
      pointHoverRadius: 6,
      showLine: false,
    }];

    // Add trend line if available
    if (trend && typeof trend.slope === 'number' && typeof trend.intercept === 'number') {
      var minX = Math.min.apply(null, x);
      var maxX = Math.max.apply(null, x);
      datasets.push({
        label: 'Trend',
        data: [
          { x: minX, y: trend.slope * minX + trend.intercept },
          { x: maxX, y: trend.slope * maxX + trend.intercept },
        ],
        type: 'line',
        borderColor: CHART_COLORS.secondary,
        borderWidth: 2,
        borderDash: [5, 5],
        pointRadius: 0,
        fill: false,
      });
    }

    var chart = new Chart(ctx, {
      type: 'scatter',
      data: { datasets: datasets },
      options: scatterLineOptions(title, 'X', 'Price (Rs)', true),
    });

    chartInstances.set(chartIdFromTile(tile), chart);
  }

  function renderLineChart(tile, slice, chartType, title) {
    var canvas = tile.querySelector('canvas');
    if (!canvas) return;

    var ctx = canvas.getContext('2d');
    destroyChart(chartIdFromTile(tile));

    var labels = slice.labels || [];
    var series = slice.series || [];

    var datasets = series.map(function(data, i) {
      return {
        label: Array.isArray(data) ? 'Series ' + (i + 1) : (labels[i] || 'Series ' + (i + 1)),
        data: Array.isArray(data) ? data : (slice.series || []),
        borderColor: i === 0 ? CHART_COLORS.primary : CHART_COLORS.secondary,
        backgroundColor: i === 0 ? CHART_COLORS.primaryLight : 'hsla(140, 60%, 40%, 0.15)',
        tension: 0.2,
        fill: false,
        pointRadius: 4,
        pointHoverRadius: 6,
      };
    });

    var chart = new Chart(ctx, {
      type: 'line',
      data: { labels: labels, datasets: datasets },
      options: scatterLineOptions(title, 'Category', 'Median Price (Rs)', false),
    });

    chartInstances.set(chartIdFromTile(tile), chart);
  }

  function renderBarChart(tile, slice, chartType, title) {
    var canvas = tile.querySelector('canvas');
    if (!canvas) return;

    var ctx = canvas.getContext('2d');
    destroyChart(chartIdFromTile(tile));

    var labels = slice.labels || [];
    var series = slice.series || [];

    // Handle both horizontal and vertical bars
    var chartId = chartIdFromTile(tile);
    var isHorizontal = chartId.indexOf('chart_08') !== -1 || chartId.indexOf('chart_13') !== -1;

    var datasets = [{
      label: 'Median Price (Rs)',
      data: series,
      backgroundColor: CHART_COLORS.primaryLight,
      borderColor: CHART_COLORS.primary,
      borderWidth: 1,
      borderRadius: 4,
    }];

    var chart = new Chart(ctx, {
      type: 'bar',
      data: { labels: labels, datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: isHorizontal ? 'y' : 'x',
        plugins: {
          title: {
            display: !!title,
            text: title,
            font: { size: 14, weight: '500' },
            color: CHART_COLORS.fg,
            padding: { bottom: 12 },
          },
          legend: { display: false },
          tooltip: {
            backgroundColor: CHART_COLORS.fg,
            titleColor: '#fff',
            bodyColor: '#fff',
            padding: 10,
            cornerRadius: 6,
            callbacks: {
              label: function(ctx) {
                var val = isHorizontal ? ctx.parsed.x : ctx.parsed.y;
                return 'Rs' + formatNumber(val);
              },
            },
          },
        },
        scales: {
          x: isHorizontal ? valueAxis('Price (Rs)') : categoryAxis(),
          y: isHorizontal ? categoryAxis() : valueAxis('Price (Rs)'),
        },
      },
    });

    chartInstances.set(chartIdFromTile(tile), chart);
  }

  function renderPieChart(tile, slice, chartType, title) {
    var canvas = tile.querySelector('canvas');
    if (!canvas) return;

    var ctx = canvas.getContext('2d');
    destroyChart(chartIdFromTile(tile));

    var labels = slice.labels || [];
    var series = slice.series || [];

    var chart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: labels,
        datasets: [{
          data: series,
          backgroundColor: [
            'hsl(210, 90%, 50%)',
            'hsl(140, 60%, 40%)',
            'hsl(40, 80%, 55%)',
            'hsl(260, 60%, 50%)',
            'hsl(0, 70%, 50%)',
            'hsl(25, 90%, 50%)',
            'hsl(180, 60%, 40%)',
            'hsl(320, 60%, 50%)',
          ],
          borderWidth: 2,
          borderColor: '#fff',
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '60%',
        plugins: {
          title: {
            display: !!title,
            text: title,
            font: { size: 14, weight: '500' },
            color: CHART_COLORS.fg,
            padding: { bottom: 12 },
          },
          legend: { position: 'right', labels: { usePointStyle: true, padding: 16, color: CHART_COLORS.muted, font: { size: 11 } } },
          tooltip: {
            backgroundColor: CHART_COLORS.fg,
            titleColor: '#fff',
            bodyColor: '#fff',
            padding: 10,
            cornerRadius: 6,
            callbacks: {
              label: function(ctx) {
                var total = series.reduce(function(a, b) { return a + b; }, 0);
                var pct = ((ctx.parsed / total) * 100).toFixed(1);
                return ctx.label + ': ' + ctx.parsed + ' listings (' + pct + '%)';
              },
            },
          },
        },
      },
    });

    chartInstances.set(chartIdFromTile(tile), chart);
  }

  function renderHeatmap(tile, slice, chartType, title) {
    var canvas = tile.querySelector('canvas');
    if (!canvas) return;

    var ctx = canvas.getContext('2d');
    destroyChart(chartIdFromTile(tile));

    var features = slice.features || [];
    var matrix = slice.matrix || [];

    // Convert matrix to Chart.js heatmap format (using bar chart with color intensity)
    // For correlation heatmap, we will render as a custom approach
    if (features.length && matrix.length) {
      renderCorrelationHeatmap(ctx, tile, features, matrix, title);
      return;
    }

    // Fallback: simple message
    ctx.font = '14px system-ui';
    ctx.fillStyle = CHART_COLORS.muted;
    ctx.textAlign = 'center';
    ctx.fillText('Correlation data unavailable', canvas.width / 2, canvas.height / 2);
  }

  function renderCorrelationHeatmap(ctx, tile, features, matrix, title) {
    // We'll use a custom canvas drawing for heatmap since Chart.js doesn't have native heatmap
    var canvas = tile.querySelector('canvas');
    var dpr = window.devicePixelRatio || 1;
    var rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    var padding = { top: 40, right: 20, bottom: 80, left: 120 };
    var w = rect.width - padding.left - padding.right;
    var h = rect.height - padding.top - padding.bottom;
    var cellW = w / features.length;
    var cellH = h / features.length;

    // Clear
    ctx.clearRect(0, 0, rect.width, rect.height);

    // Title
    ctx.font = '14px system-ui';
    ctx.fillStyle = CHART_COLORS.fg;
    ctx.textAlign = 'center';
    ctx.fillText(title, rect.width / 2, 24);

    // Draw cells
    features.forEach(function(feat, i) {
      features.forEach(function(feat2, j) {
        var val = matrix[i]?.[j] ?? 0;
        var intensity = Math.abs(val);
        var hue = val >= 0 ? 140 : 0; // Green for positive, red for negative
        ctx.fillStyle = 'hsl(' + hue + ', 70%, ' + (45 - intensity * 25) + '%)';
        ctx.fillRect(padding.left + j * cellW, padding.top + i * cellH, cellW - 1, cellH - 1);

        // Value text
        ctx.fillStyle = intensity > 0.5 ? '#fff' : CHART_COLORS.fg;
        ctx.font = '10px system-ui';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(val.toFixed(2), padding.left + j * cellW + cellW / 2, padding.top + i * cellH + cellH / 2);
      });
    });

    // Y-axis labels
    ctx.font = '10px system-ui';
    ctx.fillStyle = CHART_COLORS.fg;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    features.forEach(function(feat, i) {
      ctx.fillText(feat, padding.left - 8, padding.top + i * cellH + cellH / 2);
    });

    // X-axis labels (rotated)
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    features.forEach(function(feat, j) {
      ctx.save();
      ctx.translate(padding.left + j * cellW + cellW / 2, padding.top + h + 8);
      ctx.rotate(-Math.PI / 4);
      ctx.fillText(feat, 0, 0);
      ctx.restore();
    });
  }

  function renderWordCloud(tile, slice, chartType, title) {
    var container = tile.querySelector('.chart-wordcloud');
    if (!container) return;

    var words = slice.words || [];
    if (!words.length) {
      container.textContent = 'No word cloud data available';
      return;
    }

    // Clear container
    container.innerHTML = '';

    // Find max frequency for scaling
    var maxFreq = Math.max.apply(null, words.map(function(w) { return w.frequency || w.value || 1; }));
    var minFreq = Math.min.apply(null, words.map(function(w) { return w.frequency || w.value || 1; }));

    words.slice(0, 100).forEach(function(word) {
      var freq = word.frequency || word.value || 1;
      var ratio = (freq - minFreq) / (maxFreq - minFreq || 1);
      var size = 0.8 + ratio * 2.2; // 0.8rem to 3rem
      var opacity = 0.5 + ratio * 0.5;

      var span = document.createElement('span');
      span.className = 'wc-word';
      span.textContent = word.text || word.word || '';
      span.style.fontSize = size + 'rem';
      span.style.opacity = opacity;
      span.style.color = 'hsl(210, 90%, ' + (35 + ratio * 20) + '%)';
      span.title = (word.text || word.word) + ': ' + freq + ' occurrences';
      container.appendChild(span);
    });
  }

  // ==================== CHART OPTIONS HELPERS ====================

  function baseOptions(title) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: !!title,
          text: title,
          font: { size: 14, weight: '500' },
          color: CHART_COLORS.fg,
          padding: { bottom: 12 },
        },
        legend: {
          display: false,
          labels: { color: CHART_COLORS.muted, font: { size: 11 } },
        },
        tooltip: {
          backgroundColor: CHART_COLORS.fg,
          titleColor: CHART_COLORS.cardBg,
          bodyColor: CHART_COLORS.cardBg,
          padding: 10,
          cornerRadius: 6,
          callbacks: {
            label: function(ctx) { return (ctx.dataset.label || '') + ': Rs' + formatNumber(ctx.parsed.y ?? ctx.parsed); },
          },
        },
      },
      scales: {
        x: categoryAxis(),
        y: valueAxis('Price (Rs)'),
      },
    };
  }

  function scatterLineOptions(title, xLabel, yLabel, isScatter) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: !!title,
          text: title,
          font: { size: 14, weight: '500' },
          color: CHART_COLORS.fg,
          padding: { bottom: 12 },
        },
        legend: {
          display: true,
          position: 'top',
          labels: { color: CHART_COLORS.muted, font: { size: 11 }, usePointStyle: true },
        },
        tooltip: {
          backgroundColor: CHART_COLORS.fg,
          titleColor: '#fff',
          bodyColor: '#fff',
          padding: 10,
          cornerRadius: 6,
        },
      },
      scales: {
        x: {
          type: isScatter ? 'linear' : 'category',
          title: { display: true, text: xLabel, color: CHART_COLORS.muted, font: { size: 11 } },
          grid: { color: CHART_COLORS.grid },
          ticks: { color: CHART_COLORS.muted, font: { size: 10 }, maxTicksLimit: 8, callback: isScatter ? function(v) { return 'Rs' + formatNumber(v); } : undefined },
        },
        y: {
          type: 'linear',
          title: { display: true, text: yLabel, color: CHART_COLORS.muted, font: { size: 11 } },
          grid: { color: CHART_COLORS.grid },
          ticks: { color: CHART_COLORS.muted, font: { size: 10 }, callback: function(v) { return 'Rs' + formatNumber(v); } },
        },
      },
    };
  }

  function boxPlotOptions(title, counts) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: !!title,
          text: title,
          font: { size: 14, weight: '500' },
          color: CHART_COLORS.fg,
          padding: { bottom: 12 },
        },
        legend: { display: false },
        tooltip: {
          backgroundColor: CHART_COLORS.fg,
          titleColor: '#fff',
          bodyColor: '#fff',
          padding: 10,
          cornerRadius: 6,
          callbacks: {
            label: function(ctx) {
              var d = ctx.raw;
              if (!d || d.length < 5) return '';
              return [
                'Min: Rs' + formatNumber(d[0]),
                'Q1: Rs' + formatNumber(d[1]),
                'Median: Rs' + formatNumber(d[2]),
                'Q3: Rs' + formatNumber(d[3]),
                'Max: Rs' + formatNumber(d[4]),
                counts[ctx.dataIndex] !== undefined ? 'Count: ' + counts[ctx.dataIndex] : '',
              ].filter(Boolean).join('\n');
            },
          },
        },
      },
      scales: {
        x: categoryAxis(),
        y: valueAxis('Price (Rs)'),
      },
    };
  }

  function categoryAxis() {
    return {
      grid: { display: false },
      ticks: { color: CHART_COLORS.muted, font: { size: 10 }, maxRotation: 45, minRotation: 0 },
    };
  }

  function valueAxis(label) {
    return {
      grid: { color: CHART_COLORS.grid },
      ticks: { color: CHART_COLORS.muted, font: { size: 10 }, callback: function(v) { return 'Rs' + formatNumber(v); } },
      title: { display: true, text: label, color: CHART_COLORS.muted, font: { size: 11 } },
    };
  }

  // ==================== UTILITIES ====================

  function chartIdFromTile(tile) {
    return tile.dataset.chartId;
  }

  function destroyChart(chartId) {
    var existing = chartInstances.get(chartId);
    if (existing) {
      existing.destroy();
      chartInstances.delete(chartId);
    }
  }

  function calculateBounds(markers) {
    var minLat = Infinity, minLng = Infinity, maxLat = -Infinity, maxLng = -Infinity;
    markers.forEach(function(m) {
      if (m.lat && m.lng) {
        minLat = Math.min(minLat, m.lat);
        minLng = Math.min(minLng, m.lng);
        maxLat = Math.max(maxLat, m.lat);
        maxLng = Math.max(maxLng, m.lng);
      }
    });
    if (!isFinite(minLat)) return [[20, 68], [29, 97]]; // India fallback
    var pad = 0.05;
    return [[minLat - pad, minLng - pad], [maxLat + pad, maxLng + pad]];
  }

  function formatNumber(n) {
    if (n >= 1e7) return (n / 1e7).toFixed(1) + 'Cr';
    if (n >= 1e5) return (n / 1e5).toFixed(1) + 'L';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return Math.round(n).toLocaleString();
  }

  // Expose for debugging
  window.__analyticsDebug = { chartInstances: chartInstances, spatialMap: spatialMap };
})();