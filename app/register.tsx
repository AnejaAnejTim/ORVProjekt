import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import {
  Alert,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  useColorScheme,
  View,
} from 'react-native';

const Register = (): React.JSX.Element => {
  const isDarkMode = useColorScheme() === 'dark';
  const router = useRouter();

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');

  const handleRegister = async () => {
    const isValidEmail = (email: string) => {
      const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      return regex.test(email);
    };

    if (!isValidEmail(email)) {
      setError('Prosim vnesite veljaven email naslov.');
      return;
    }

    try {
      const res = await fetch('http://100.117.101.70:3001/users', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email,
          username,
          password,
        }),
      });

      const data = await res.json();

      if (res.status === 409) {
        setError('Uporabnik že obstaja');
      } else if (res.ok && data._id) {
        Alert.alert('Registracija uspešna', 'Sedaj se lahko prijavite.');
        router.push('/login');
      } else {
        setUsername('');
        setPassword('');
        setEmail('');
        setError(data.message || 'Registracija ni uspela');
      }
    } catch (err) {
      setError('Napaka pri registraciji');
      console.error(err);
    }
  };

  return (
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
        <Text style={[styles.title, { color: '#b0d16b' }]}>Registracija</Text>

        <TextInput
          placeholder="Email"
          placeholderTextColor={isDarkMode ? '#aaa' : '#666'}
          style={[
            styles.input,
            {
              color: isDarkMode ? 'white' : 'black',
              borderColor: isDarkMode ? '#555' : '#ccc',
            },
          ]}
          value={email}
          onChangeText={setEmail}
          keyboardType="email-address"
          autoCapitalize="none"
        />

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
          style={[
            styles.button,
            !(email && username && password) && styles.disabled,
          ]}
          onPress={handleRegister}
          disabled={!(email && username && password)}
        >
          <Text style={styles.buttonText}>Registracija</Text>
        </TouchableOpacity>

        <TouchableOpacity onPress={() => router.push('/login')}>
          <Text
            style={[
              styles.loginLink,
              { color: isDarkMode ? '#66ccff' : '#003366' },
            ]}
          >
            Že imate račun? Prijavite se tukaj
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

export default Register;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: '5%',
    justifyContent: 'center',
    alignItems: 'center',
  },
  card: {
    width: '100%',
    maxWidth: 400,
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
  },
  shadowLight: {
    shadowColor: '#000',
    shadowOpacity: 0.1,
    shadowRadius: 10,
    elevation: 5,
  },
  shadowDark: {
    shadowColor: '#000',
    shadowOpacity: 0.4,
    shadowRadius: 12,
    elevation: 10,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  input: {
    width: '100%',
    borderWidth: 1,
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
  },
  button: {
    backgroundColor: '#b0d16b',
    padding: 12,
    borderRadius: 8,
    width: '100%',
    alignItems: 'center',
    marginBottom: 12,
  },
  disabled: {
    backgroundColor: '#ccc',
  },
  buttonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
  },
  loginLink: {
    fontSize: 14,
    textAlign: 'center',
  },
  error: {
    color: 'red',
    marginBottom: 12,
    textAlign: 'center',
  },
});
