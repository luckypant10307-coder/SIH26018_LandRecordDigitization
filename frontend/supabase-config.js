/**
 * Supabase Configuration
 * Replace these with your actual Supabase project credentials
 */

const SUPABASE_CONFIG = {
  // Get these from your Supabase project settings
  // https://app.supabase.com/project/YOUR_PROJECT/settings/api
  url: 'https://qyvdjgiaypcdywrrdlwu.supabase.co',
  anonKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InF5dmRqZ2lheXBjZHl3cnJkbHd1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODg1MzgxMjIsImV4cCI6MjEwNDExNDEyMn0.hYlIGE8ppp5X-NVz-plf70P-_Svz-pkLObvAioS4j_o'
};

// Initialize Supabase client only when window.supabase is available
let supabaseClient = null;

// Function to initialize Supabase
function initSupabase() {
  if (!window.supabase) {
    console.error('Supabase library not loaded yet');
    return false;
  }

  if (!supabaseClient) {
    supabaseClient = window.supabase.createClient(
      SUPABASE_CONFIG.url,
      SUPABASE_CONFIG.anonKey
    );
    console.log('Supabase client initialized');
  }
  return true;
}

// Auth helper functions
const SupabaseAuth = {
  // Sign up new user
  async signUp(email, password, userData = {}) {
    if (!initSupabase()) throw new Error('Supabase not initialized');

    const { data, error } = await supabaseClient.auth.signUp({
      email,
      password,
      options: {
        data: userData // Store role and other metadata
      }
    });

    if (error) throw error;
    return data;
  },

  // Sign in existing user
  async signIn(email, password) {
    if (!initSupabase()) throw new Error('Supabase not initialized');

    const { data, error } = await supabaseClient.auth.signInWithPassword({
      email,
      password
    });

    if (error) throw error;
    return data;
  },

  // Sign out
  async signOut() {
    if (!initSupabase()) throw new Error('Supabase not initialized');

    const { error } = await supabaseClient.auth.signOut();
    if (error) throw error;
  },

  // Get current session
  async getSession() {
    if (!initSupabase()) throw new Error('Supabase not initialized');

    const { data, error } = await supabaseClient.auth.getSession();
    if (error) throw error;
    return data.session;
  },

  // Get current user
  async getUser() {
    if (!initSupabase()) throw new Error('Supabase not initialized');

    const { data, error } = await supabaseClient.auth.getUser();
    if (error) throw error;
    return data.user;
  },

  // Listen to auth state changes
  onAuthStateChange(callback) {
    if (!initSupabase()) throw new Error('Supabase not initialized');

    return supabaseClient.auth.onAuthStateChange(callback);
  },

  // Update user metadata (role, etc.)
  async updateUserMetadata(updates) {
    if (!initSupabase()) throw new Error('Supabase not initialized');

    const { data, error } = await supabaseClient.auth.updateUser({
      data: updates
    });

    if (error) throw error;
    return data;
  },

  // Reset password
  async resetPassword(email) {
    if (!initSupabase()) throw new Error('Supabase not initialized');

    const { data, error } = await supabaseClient.auth.resetPasswordForEmail(email, {
      redirectTo: `${window.location.origin}/reset-password.html`
    });

    if (error) throw error;
    return data;
  },

  // Update password
  async updatePassword(newPassword) {
    if (!initSupabase()) throw new Error('Supabase not initialized');

    const { data, error } = await supabaseClient.auth.updateUser({
      password: newPassword
    });

    if (error) throw error;
    return data;
  }
};

// Export for use in other files
window.SupabaseAuth = SupabaseAuth;
window.supabaseClient = supabaseClient;
window.initSupabase = initSupabase;
