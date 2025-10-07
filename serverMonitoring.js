#!/usr/bin/env node
/**
 * Server Monitoring Service
 * Tracks request metrics, error rates, response times, and uptime
 * Provides daily statistics and real-time status
 */

class ServerMonitoring {
  constructor() {
    this.startTime = Date.now();
    this.metrics = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      totalResponseTime: 0,
      slowestRequest: 0,
      fastestRequest: Infinity,
      endpointStats: {}, // { '/path': { count, errors, avgTime } }
      errorsByCode: {}, // { '500': 10, '404': 5 }
      dailyStats: {}, // { 'YYYY-MM-DD': { requests, errors, avgTime } }
      recentErrors: [], // Last 100 errors with details
      hourlyRequests: new Array(24).fill(0), // Request count per hour of day
    };
    this.currentHour = new Date().getHours();
  }

  /**
   * Get current date in YYYY-MM-DD format
   */
  getTodayKey() {
    const now = new Date();
    return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
  }

  /**
   * Record a request
   */
  recordRequest({ endpoint, statusCode, responseTime, error, method }) {
    this.metrics.totalRequests++;
    
    const isSuccess = statusCode >= 200 && statusCode < 400;
    if (isSuccess) {
      this.metrics.successfulRequests++;
    } else {
      this.metrics.failedRequests++;
    }

    // Track response time
    if (responseTime) {
      this.metrics.totalResponseTime += responseTime;
      this.metrics.slowestRequest = Math.max(this.metrics.slowestRequest, responseTime);
      this.metrics.fastestRequest = Math.min(this.metrics.fastestRequest, responseTime);
    }

    // Track endpoint stats
    if (!this.metrics.endpointStats[endpoint]) {
      this.metrics.endpointStats[endpoint] = {
        count: 0,
        errors: 0,
        totalTime: 0,
        methods: {},
      };
    }
    const epStats = this.metrics.endpointStats[endpoint];
    epStats.count++;
    if (!isSuccess) epStats.errors++;
    if (responseTime) epStats.totalTime += responseTime;
    epStats.methods[method] = (epStats.methods[method] || 0) + 1;

    // Track errors by status code
    if (!isSuccess) {
      const code = String(statusCode);
      this.metrics.errorsByCode[code] = (this.metrics.errorsByCode[code] || 0) + 1;

      // Store recent errors (limit to 100)
      this.metrics.recentErrors.unshift({
        timestamp: Date.now(),
        endpoint,
        statusCode,
        method,
        error: error || 'Unknown error',
      });
      if (this.metrics.recentErrors.length > 100) {
        this.metrics.recentErrors.pop();
      }
    }

    // Track daily stats
    const today = this.getTodayKey();
    if (!this.metrics.dailyStats[today]) {
      this.metrics.dailyStats[today] = {
        requests: 0,
        errors: 0,
        totalTime: 0,
      };
    }
    const dailyStats = this.metrics.dailyStats[today];
    dailyStats.requests++;
    if (!isSuccess) dailyStats.errors++;
    if (responseTime) dailyStats.totalTime += responseTime;

    // Track hourly distribution
    const hour = new Date().getHours();
    if (hour !== this.currentHour) {
      this.currentHour = hour;
    }
    this.metrics.hourlyRequests[hour]++;

    // Clean up old daily stats (keep last 30 days)
    this.cleanupOldStats();
  }

  /**
   * Remove stats older than 30 days
   */
  cleanupOldStats() {
    const thirtyDaysAgo = Date.now() - (30 * 24 * 60 * 60 * 1000);
    const cutoffDate = new Date(thirtyDaysAgo);
    const cutoffKey = `${cutoffDate.getFullYear()}-${String(cutoffDate.getMonth() + 1).padStart(2, '0')}-${String(cutoffDate.getDate()).padStart(2, '0')}`;

    for (const key in this.metrics.dailyStats) {
      if (key < cutoffKey) {
        delete this.metrics.dailyStats[key];
      }
    }
  }

  /**
   * Get comprehensive status report
   */
  getStatus() {
    const uptime = Date.now() - this.startTime;
    const uptimeSeconds = Math.floor(uptime / 1000);
    const uptimeMinutes = Math.floor(uptimeSeconds / 60);
    const uptimeHours = Math.floor(uptimeMinutes / 60);
    const uptimeDays = Math.floor(uptimeHours / 24);

    const avgResponseTime = this.metrics.totalRequests > 0
      ? Math.round(this.metrics.totalResponseTime / this.metrics.totalRequests)
      : 0;

    const today = this.getTodayKey();
    const todayStats = this.metrics.dailyStats[today] || { requests: 0, errors: 0, totalTime: 0 };
    const todayErrorRate = todayStats.requests > 0
      ? ((todayStats.errors / todayStats.requests) * 100).toFixed(2)
      : '0.00';
    const todayAvgTime = todayStats.requests > 0
      ? Math.round(todayStats.totalTime / todayStats.requests)
      : 0;

    // Calculate top endpoints
    const topEndpoints = Object.entries(this.metrics.endpointStats)
      .map(([endpoint, stats]) => ({
        endpoint,
        count: stats.count,
        errors: stats.errors,
        avgTime: stats.count > 0 ? Math.round(stats.totalTime / stats.count) : 0,
        errorRate: stats.count > 0 ? ((stats.errors / stats.count) * 100).toFixed(2) : '0.00',
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);

    return {
      uptime: {
        ms: uptime,
        seconds: uptimeSeconds,
        minutes: uptimeMinutes,
        hours: uptimeHours,
        days: uptimeDays,
        formatted: `${uptimeDays}d ${uptimeHours % 24}h ${uptimeMinutes % 60}m ${uptimeSeconds % 60}s`,
      },
      requests: {
        total: this.metrics.totalRequests,
        successful: this.metrics.successfulRequests,
        failed: this.metrics.failedRequests,
        successRate: this.metrics.totalRequests > 0
          ? ((this.metrics.successfulRequests / this.metrics.totalRequests) * 100).toFixed(2)
          : '100.00',
      },
      performance: {
        avgResponseTime,
        fastestRequest: this.metrics.fastestRequest === Infinity ? 0 : this.metrics.fastestRequest,
        slowestRequest: this.metrics.slowestRequest,
      },
      today: {
        date: today,
        requests: todayStats.requests,
        errors: todayStats.errors,
        successful: todayStats.requests - todayStats.errors,
        errorRate: todayErrorRate,
        avgResponseTime: todayAvgTime,
      },
      topEndpoints,
      errorsByCode: this.metrics.errorsByCode,
      recentErrors: this.metrics.recentErrors.slice(0, 10), // Last 10 errors
      hourlyDistribution: this.metrics.hourlyRequests,
    };
  }

  /**
   * Get simple health status
   */
  getHealth() {
    const errorRate = this.metrics.totalRequests > 0
      ? (this.metrics.failedRequests / this.metrics.totalRequests) * 100
      : 0;

    let status = 'healthy';
    if (errorRate > 50) status = 'critical';
    else if (errorRate > 20) status = 'degraded';
    else if (errorRate > 5) status = 'warning';

    return {
      status,
      uptime: Date.now() - this.startTime,
      errorRate: errorRate.toFixed(2),
      timestamp: Date.now(),
    };
  }

  /**
   * Reset all metrics (useful for testing)
   */
  reset() {
    this.startTime = Date.now();
    this.metrics = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      totalResponseTime: 0,
      slowestRequest: 0,
      fastestRequest: Infinity,
      endpointStats: {},
      errorsByCode: {},
      dailyStats: {},
      recentErrors: [],
      hourlyRequests: new Array(24).fill(0),
    };
  }
}

// Singleton instance
const monitoring = new ServerMonitoring();

export default monitoring;
