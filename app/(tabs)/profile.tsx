import { StyleSheet, TouchableOpacity, View } from 'react-native';

import ParallaxScrollView from '@/components/ParallaxScrollView';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { IconSymbol } from '@/components/ui/IconSymbol';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useRouter } from 'expo-router';
import React, { useContext } from 'react';
import { UserContext } from '../userContext';

export default function ProfileScreen() {
  const { user, setUser, refreshUser } = useContext(UserContext);
  const router = useRouter();

  const handleLogout = async () => {
    await AsyncStorage.removeItem('token');
    setUser(null);
    await refreshUser();
    router.replace('/login');
  };


  return (
    <ParallaxScrollView
      headerBackgroundColor={{ light: '#D0D0D0', dark: '#353636' }}
      headerImage={
        <IconSymbol
          size={310}
          color="#808080"
          name="person.crop.circle"
          style={styles.headerImage}
        />
      }>
      <ThemedView style={styles.titleContainer}>
        <ThemedText type="title">Profil</ThemedText>
      </ThemedView>

      <ThemedText>
        Zdravo {user?.username ? user.username : ''}!
      </ThemedText>

      {/* Logout button */}
      <View style={styles.logoutContainer}>
        <TouchableOpacity onPress={handleLogout} style={styles.logoutButton}>
          <ThemedText style={styles.logoutText}>Logout</ThemedText>
        </TouchableOpacity>
      </View>
    </ParallaxScrollView>

  );
}

const styles = StyleSheet.create({
  headerImage: {
    color: '#808080',
    bottom: -90,
    left: -35,
    position: 'absolute',
  },
  titleContainer: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 20,
  },
  logoutContainer: {
    marginTop: 40,
    alignItems: 'center',
  },
  logoutButton: {
    backgroundColor: '#b22222', // red button
    paddingVertical: 12,
    paddingHorizontal: 30,
    borderRadius: 8,
  },
  logoutText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
  },
});
