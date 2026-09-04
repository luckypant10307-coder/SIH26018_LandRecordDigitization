/**
 * Supabase Configuration
 * Replace these with your actual Supabase project credentials
 */

const SUPABASE_CONFIG = {
  // Get these from your Supabase project settings
  // https://app.supabase.com/project/YOUR_PROJECT/settings/api
  url: 'https://rpjhtegnbsvjgofmntdc.supabase.co',
  anonKey: 'sb_publishable_q4PdYTtYfJZCHB9eGLK8rA_QrO5lKxu'
};

// Initialize Supabase client only when window.supabase is available
let supabase = null;

// Function to initialize Supabase
function initSupabase() {
  if (!window.supabase) {
    console.error('Supabase library not loaded yet');
    return false;
  }

  if (!supabase) {
    supabase = window.supabase.createClient(
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

    const { data, error } = await supabase.auth.signUp({
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

    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password
    });

    if (error) throw error;
    return data;
  },

  // Sign out
  async signOut() {
    if (!initSupabase()) throw new Error('Supabase not initialized');

    const { error } = await supabase.auth.signOut();
    if (error) throw error;
  },

  // Get current session
  async getSession() {
    if (!initSupabase()) throw new Error('Supabase not initialized');

    const { data, error } = await supabase.auth.getSession();
    if (error) throw error;
    return data.session;
  },

  // Get current user
  async getUser() {
    if (!initSupabase()) throw new Error('Supabase not initialized');

    const { data, error } = await supabase.auth.getUser();
    if (error) throw error;
    return data.user;
  },

  // Listen to auth state changes
  onAuthStateChange(callback) {
    if (!initSupabase()) throw new Error('Supabase not initialized');

    return supabase.auth.onAuthStateChange(callback);
  },

  // Update user metadata (role, etc.)
  async updateUserMetadata(updates) {
    if (!initSupabase()) throw new Error('Supabase not initialized');

    const { data, error } = await supabase.auth.updateUser({
      data: updates
    });

    if (error) throw error;
    return data;
  },

  // Reset password
  async resetPassword(email) {
    if (!initSupabase()) throw new Error('Supabase not initialized');

    const { data, error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: `${window.location.origin}/reset-password.html`
    });

    if (error) throw error;
    return data;
  },

  // Update password
  async updatePassword(newPassword) {
    if (!initSupabase()) throw new Error('Supabase not initialized');

    const { data, error } = await supabase.auth.updateUser({
      password: newPassword
    });

    if (error) throw error;
    return data;
  }
};

// Export for use in other files
window.SupabaseAuth = SupabaseAuth;
window.supabaseClient = supabase;
window.initSupabase = initSupabase;
