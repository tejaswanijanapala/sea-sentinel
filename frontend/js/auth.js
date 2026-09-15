/**
 * Sea Sentinel — RBAC & Authentication Manager
 * ============================================
 * Manages user authentication, token storage, role-based UI scoping (ADMIN vs USER),
 * and automatic dashboard re-rendering upon role transitions.
 */

class AuthManager {
  constructor() {
    this.tokenKey = "sea_sentinel_auth_token";
    this.profileKey = "sea_sentinel_user_profile";
    this.roleKey = "sea_sentinel_active_role";
    this.subscribers = [];
    
    // Default seed accounts
    this.accounts = {
      ADMIN: {
        email: "admin@seasentinel.ocean",
        password: "admin123",
        role: "ADMIN",
        title: "Chief Hydrographer & ML Admin",
        badge: "ADMINISTRATOR"
      },
      USER: {
        email: "operator@seasentinel.ocean",
        password: "user123",
        role: "USER",
        title: "Surveillance Sonar Operator",
        badge: "OPERATOR"
      }
    };

    this.currentUser = null;
    this.token = null;
    this.init();
  }

  init() {
    try {
      const savedToken = localStorage.getItem(this.tokenKey);
      const savedProfile = localStorage.getItem(this.profileKey);
      
      if (savedToken && savedProfile) {
        this.token = savedToken;
        this.currentUser = JSON.parse(savedProfile);
      } else {
        // Default to USER role (Operator) on initial boot
        this._setLocalFallbackUser("USER");
      }
    } catch (e) {
      console.warn("Auth initialization error, resetting to fallback user:", e);
      this._setLocalFallbackUser("USER");
    }

    // Apply UI state once DOM is loaded
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", () => this.applyRoleToUI());
    } else {
      this.applyRoleToUI();
    }
  }

  _setLocalFallbackUser(roleName = "USER") {
    const role = roleName.toUpperCase() === "ADMIN" ? "ADMIN" : "USER";
    const acc = this.accounts[role];
    this.currentUser = {
      user_id: role === "ADMIN" ? "usr_admin_001" : "usr_operator_002",
      email: acc.email,
      full_name: acc.title,
      role: role,
      organization: role === "ADMIN" ? "Sea Sentinel Ocean Command" : "Maritime Patrol Unit 4",
      avatar_initials: role === "ADMIN" ? "AD" : "SO",
      permissions: role === "ADMIN"
        ? ["global_ocean_map", "ai_models_manage", "model_evaluation_run", "system_telemetry", "global_metrics", "admin_settings"]
        : ["current_input_gis", "debris_inspection", "risk_analysis", "input_metrics", "report_generation"]
    };
    // Dummy JWT structure: base64(payload).signature
    const payload = btoa(JSON.stringify({ sub: this.currentUser.user_id, email: this.currentUser.email, role: role, exp: Date.now() + 86400000 }));
    this.token = `${payload}.signature_offline_dev`;
    this._persist();
  }

  _persist() {
    try {
      if (this.token) localStorage.setItem(this.tokenKey, this.token);
      if (this.currentUser) {
        localStorage.setItem(this.profileKey, JSON.stringify(this.currentUser));
        localStorage.setItem(this.roleKey, this.currentUser.role);
      }
    } catch (e) {
      console.warn("Error saving auth state to localStorage:", e);
    }
  }

  getRole() {
    return this.currentUser ? this.currentUser.role : "USER";
  }

  isAdmin() {
    return this.getRole() === "ADMIN";
  }

  isUser() {
    return this.getRole() === "USER";
  }

  getUser() {
    return this.currentUser;
  }

  getToken() {
    return this.token;
  }

  getAuthHeader() {
    return this.token ? { "Authorization": `Bearer ${this.token}` } : {};
  }

  onRoleChange(cb) {
    if (typeof cb === "function") {
      this.subscribers.push(cb);
    }
  }

  _notifySubscribers() {
    const role = this.getRole();
    const user = this.getUser();
    this.subscribers.forEach(cb => {
      try { cb(role, user); } catch (e) { console.error("Error in auth subscriber:", e); }
    });
  }

  async login(email, password) {
    try {
      const res = await fetch("http://localhost:8000/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password })
      });

      if (res.ok) {
        const data = await res.json();
        this.token = data.access_token;
        this.currentUser = data.user;
        this._persist();
        this.applyRoleToUI();
        this._notifySubscribers();
        return { success: true, user: this.currentUser };
      } else {
        const err = await res.json().catch(() => ({ detail: "Login failed" }));
        return { success: false, error: err.detail || "Authentication failed." };
      }
    } catch (e) {
      // Offline fallback login check
      const cleanEmail = (email || "").trim().toLowerCase();
      if ((cleanEmail === "admin@seasentinel.ocean" || cleanEmail === "admin") && password === "admin123") {
        this._setLocalFallbackUser("ADMIN");
        this.applyRoleToUI();
        this._notifySubscribers();
        return { success: true, user: this.currentUser, offline: true };
      } else if ((cleanEmail === "operator@seasentinel.ocean" || cleanEmail === "user" || cleanEmail === "operator") && password === "user123") {
        this._setLocalFallbackUser("USER");
        this.applyRoleToUI();
        this._notifySubscribers();
        return { success: true, user: this.currentUser, offline: true };
      }
      return { success: false, error: "Network error and unrecognized credentials." };
    }
  }

  async switchRole(targetRole) {
    const role = targetRole.toUpperCase() === "ADMIN" ? "ADMIN" : "USER";
    try {
      const res = await fetch("http://localhost:8000/api/auth/switch-role", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...this.getAuthHeader()
        },
        body: JSON.stringify({ target_role: role })
      });

      if (res.ok) {
        const data = await res.json();
        this.token = data.access_token;
        this.currentUser = data.user;
        this._persist();
      } else {
        this._setLocalFallbackUser(role);
      }
    } catch (e) {
      this._setLocalFallbackUser(role);
    }

    this.applyRoleToUI();
    this._notifySubscribers();
    return this.currentUser;
  }

  logout() {
    this._setLocalFallbackUser("USER");
    this.applyRoleToUI();
    this._notifySubscribers();
  }

  applyRoleToUI() {
    const role = this.getRole();
    const isAdmin = role === "ADMIN";
    const user = this.getUser() || {};

    // 0. Toggle body class for instant CSS rules
    document.body.classList.remove("role-admin", "role-user");
    document.body.classList.add(isAdmin ? "role-admin" : "role-user");

    // 1. Update Profile Badges & User Information in UI
    const roleBadge = document.getElementById("navUserRoleBadge");
    if (roleBadge) {
      roleBadge.textContent = isAdmin ? "ADMIN" : "OPERATOR";
      roleBadge.className = `role-badge ${isAdmin ? "badge-admin" : "badge-user"}`;
    }

    const userNameEl = document.getElementById("navUserName");
    if (userNameEl) {
      userNameEl.textContent = user.full_name || (isAdmin ? "ML Admin" : "Sonar Operator");
    }

    const userEmailEl = document.getElementById("navUserEmail");
    if (userEmailEl) {
      userEmailEl.textContent = user.email || (isAdmin ? "admin@seasentinel.ocean" : "operator@seasentinel.ocean");
    }

    const userAvatarEl = document.getElementById("navUserAvatar");
    if (userAvatarEl) {
      userAvatarEl.textContent = user.avatar_initials || (isAdmin ? "AD" : "SO");
      userAvatarEl.style.background = isAdmin 
        ? "linear-gradient(135deg, #ef4444, #f59e0b)" 
        : "linear-gradient(135deg, #00f0ff, #0284c7)";
    }

    // 2. Control Dynamic Nav & Sidebar Visibility
    // Admin elements have class 'role-admin-only', user elements 'role-user-only'
    document.querySelectorAll(".role-admin-only").forEach(el => {
      el.style.display = isAdmin ? "" : "none";
    });

    document.querySelectorAll(".role-user-only").forEach(el => {
      el.style.display = !isAdmin ? "" : "none";
    });

    // 3. Dynamic Section Header Title & Subtitle Scoping
    const gisSectionHeader = document.getElementById("gisSectionTitle");
    const gisSectionSub = document.getElementById("gisSectionSubtitle");
    if (gisSectionHeader) {
      gisSectionHeader.textContent = isAdmin ? "Entire Ocean Map (Global Repository)" : "Current Input GIS (Tactical Survey)";
    }
    if (gisSectionSub) {
      gisSectionSub.textContent = isAdmin
        ? "Multi-mission geospatial intelligence with all historical detections, DBSCAN clusters, and cross-survey tracks."
        : "Isolated spatial view of the active Side-Scan Sonar survey with verified debris coordinates and swath polygon.";
    }

    // 4. Update Scoped Metric Labels
    const metricsScopeLabel = document.getElementById("metricsScopeLabel");
    if (metricsScopeLabel) {
      metricsScopeLabel.textContent = isAdmin ? "GLOBAL DATASET METRICS" : "CURRENT SURVEY METRICS";
    }

    // 5. If in User role and currently on an admin-restricted tab, auto-switch to Dashboard / Current GIS
    if (!isAdmin) {
      const activeTab = document.querySelector(".nav-item.active, .tab-btn.active");
      const restrictedTabs = ["modelsTab", "modelPerfTab", "adminSettingsTab", "entireOceanMapTab"];
      if (activeTab && restrictedTabs.includes(activeTab.id)) {
        const defaultTab = document.getElementById("dashboardTab") || document.querySelector(".nav-item");
        if (defaultTab) defaultTab.click();
      }
    }
  }
}

// Global Auth Singleton Instance
window.authManager = new AuthManager();
