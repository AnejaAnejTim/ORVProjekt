import { useRouter } from 'expo-router';
import React, { useContext, useState } from 'react';


import {
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  useColorScheme,
  View,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  ScrollView
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { UserContext } from './userContext';
export default function Login() {
  const context = useContext(UserContext);
  if (!context) {
    throw new Error("Login must be used within a UserProvider");
  }
  const { setUser, refreshUser } = context;

  const isDarkMode = useColorScheme() === 'dark';
  const router = useRouter();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

const handleLogin = async () => {
  setError('');
  setLoading(true);
  try {
    console.log('Attempting login with:', { username, password });
    const res = await fetch('http://192.168.0.13:3001/users/appLogin', {
      method: 'POST',   
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });

    console.log('Login response status:', res.status);
    const data = await res.json();
    console.log('Login response data:', data);
    if (res.status === 200 && data.token && data.user) {
      await AsyncStorage.setItem('token', data.token);
      await AsyncStorage.setItem('email', data.user.email);
      console.log('Token saved to AsyncStorage:', data.token);
      setUser(data.user);
      await refreshUser();
      router.replace('/');
    } else {
      setError(data.message || 'Napačno uporabniško ime ali geslo');
      console.log('Login failed:', data);
    }
  } catch (err) {
    console.error('Login error:', err);
    setError('Napaka pri povezavi s strežnikom');
  } finally {
    setLoading(false);
  }
};
  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      style={{ flex: 1 }}
    >
      <ScrollView contentContainerStyle={{ flexGrow: 1 }}>
        <View
          style={[
            styles.container,
            { backgroundColor: isDarkMode ? '#121212' : '#f9fafb' },
          ]}
        >
          <View
            style={[
              styles.card,
              { backgroundColor: isDarkMode ? '#333' : '#fff' },
              isDarkMode ? styles.shadowDark : styles.shadowLight,
            ]}
          >
            <Text style={[styles.title, { color: '#b0d16b' }]}>Prijava</Text>
            <TextInput
          placeholder="Uporabniško ime"
          placeholderTextColor={isDarkMode ? '#aaa' : '#666'}
          style={[
            styles.input,
            {
              color: isDarkMode ? 'white' : 'black',
              borderColor: isDarkMode ? '#555' : '#ccc',
            },
          ]}
          value={username}
          onChangeText={setUsername}
          autoCapitalize="none"
        />
        <TextInput
            placeholder="Geslo"
            placeholderTextColor={isDarkMode ? '#aaa' : '#666'}
            style={[
              styles.input,
              {
                color: isDarkMode ? 'white' : 'black',
                borderColor: isDarkMode ? '#555' : '#ccc',
              },
            ]}
            value={password}
            onChangeText={setPassword}
            secureTextEntry
          />
          {error ? <Text style={styles.error}>{error}</Text> : null}
          <TouchableOpacity
            style={[styles.button, (!(username && password) || loading) && styles.disabled]}
            onPress={handleLogin}
            disabled={!(username && password) || loading}
          >
            {loading ? (
              <ActivityIndicator color="white" />
            ) : (
              <Text style={styles.buttonText}>Prijava</Text>
            )}
          </TouchableOpacity>
          <TouchableOpacity onPress={() => router.push('/register')} disabled={loading}>
            <Text
              style={[
                styles.linkText,
                { color: isDarkMode ? '#b0d16b' : '#6b8e23', marginTop: 15 },
              ]}
            >
              Še nimate računa? Registrirajte se
            </Text>
          </TouchableOpacity>
        </View>
      </View>
    </ScrollView>
  </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  card: {
    width: '100%',
    maxWidth: 400,
    padding: 30,
    borderRadius: 20,
  },
  shadowLight: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 10,
    elevation: 5,
  },
shadowDark: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
    elevation: 10,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 25,
    textAlign: 'center',
  },
  input: {
    height: 50,
    borderWidth: 1,
    borderRadius: 10,
    paddingHorizontal: 15,
    marginBottom: 15,
    fontSize: 16,
  },
  error: {
    color: '#ff4444',
    marginBottom: 15,
    textAlign: 'center',
  },
  button: {
    backgroundColor: '#b0d16b',
    height: 50,
    borderRadius: 10,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 10,
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: '600',
  },
  disabled: {
    opacity: 0.6,
  },
  linkText: {
    textAlign: 'center',
    fontSize: 14,
  },
});